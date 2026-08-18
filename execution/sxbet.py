"""SX Bet read-only adapter → de-vigged live fair probability.

SX Bet is an on-chain betting *exchange* (https://api.sx.bet). Reads are public
(no key). Because it's an exchange with a two-sided order book, the best-back
implied prices on both outcomes de-vig to a near-margin-free fair probability —
and the book is live/in-play, so this is a real live signal (unlike a pre-match
line). Order placement would require EIP-712 wallet signing + a funded SX-Network
wallet; that is intentionally NOT implemented here (reads only).

    sx = SXBetClient()
    q = sx.quote("Jannik Sinner", "Alexander Zverev")
    #  -> {"fair_p1": 0.80, "score": "1-1", "live": True, "league": "ATP Wimbledon", ...}
"""

import os
import secrets
import time
from typing import Optional

import requests

from execution.polymarket import _norm, _surname  # reuse name normalization

# Endpoint is env-overridable so you can point at the SX TESTNET
# (SXBET_API=https://api.toronto.sx.bet) and validate real order signing +
# posting with zero real money before touching mainnet. chainId / fill-hasher /
# USDC are auto-detected from GET /metadata for whichever API is configured, so
# switching networks only needs SXBET_API. Defaults are SX mainnet.
API = os.getenv("SXBET_API", "https://api.sx.bet").rstrip("/")
DOMAIN_VERSION = os.getenv("SXBET_DOMAIN_VERSION", "6.0")
ZERO_ADDR = "0x0000000000000000000000000000000000000000"
ZERO_BYTES32 = "0x" + "00" * 32

TENNIS_SPORT_ID = 6
MATCH_WINNER_TYPE = 52          # SX market type for match/moneyline winner

# Other live SX tennis market types (verified against /markets/active):
SET_WINNER_TYPES = {202: 1, 203: 2, 204: 3}   # "Nth Period" winner → set number
GAME_SPREAD_TYPE = 201          # game handicap, line on team one (e.g. -2.5)
TOTAL_GAMES_TYPE = 166          # total games O/U (outcome one = Over)
SET_SPREAD_TYPE = 866           # set handicap (-1.5 sets on team one)
TOTAL_SETS_TYPE = 165           # total sets O/U (2.5 in best-of-3)
_MARKETS_TTL = 30               # seconds
_QUOTE_TTL = 15                 # seconds per market order book


class SXBetClient:
    def __init__(self, session: Optional[requests.Session] = None):
        self.http = session or requests.Session()
        self.http.headers.update({"User-Agent": "tennis10-sxbet/1.0"})
        api_key = os.getenv("SXBET_API_KEY")
        if api_key:
            self.http.headers.update({"X-Api-Key": api_key})
        self._markets: dict = {}     # frozenset(surnames) -> market dict
        self._markets_at = 0.0
        self._quote_cache: dict = {}  # marketHash -> (ts, result)
        self._meta: dict = {}         # chainId / verifyingContract / baseToken

    # ── network config (auto from /metadata, env-overridable) ─────────────────

    def _meta_config(self) -> dict:
        if self._meta:
            return self._meta
        md = (self._get("/metadata") or {}).get("data", {}) or {}
        addresses = md.get("addresses") or {}
        chain_id = int(next(iter(addresses), 0)) or int(os.getenv("SXBET_CHAIN_ID", "4162"))
        usdc = (addresses.get(str(chain_id), {}) or {}).get("USDC")
        self._meta = {
            "chain_id": int(os.getenv("SXBET_CHAIN_ID", str(chain_id))),
            "verifying_contract": os.getenv(
                "SXBET_VERIFYING_CONTRACT",
                md.get("EIP712FillHasher", "0x845a2Da2D70fEDe8474b1C8518200798c60aC364")),
            "base_token": os.getenv(
                "SXBET_BASE_TOKEN",
                usdc or "0x6629Ce1Cf35Cc1329ebB4F63202F3f197b3F050B"),
        }
        return self._meta

    # ── low level ────────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None):
        try:
            r = self.http.get(f"{API}{path}", params=params or {}, timeout=15)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            return None

    def _refresh_markets(self) -> None:
        if (time.time() - self._markets_at) < _MARKETS_TTL and self._markets:
            return
        data = self._get("/markets/active", {"sportIds": TENNIS_SPORT_ID})
        self._markets_at = time.time()
        if not data:
            return
        markets = (data.get("data") or {}).get("markets") or []
        index = {}
        all_index: dict = {}
        for m in markets:
            s1, s2 = _surname(m.get("teamOneName", "")), _surname(m.get("teamTwoName", ""))
            if not s1 or not s2 or s1 == s2:
                continue
            key = frozenset((s1, s2))
            all_index.setdefault(key, {}).setdefault(m.get("type"), []).append(m)
            if m.get("type") == MATCH_WINNER_TYPE:
                index[key] = m
        if index:
            self._markets = index
        if all_index:
            self._all_markets = all_index

    def markets_for(self, player1: str, player2: str) -> dict:
        """ALL active SX markets for a fixture, keyed by market type id →
        list of market dicts (a type can list several lines). {} if none."""
        self._refresh_markets()
        s1, s2 = _surname(player1), _surname(player2)
        if not s1 or not s2:
            return {}
        return getattr(self, "_all_markets", {}).get(frozenset((s1, s2)), {})

    def taker_prices(self, market: dict) -> Optional[dict]:
        """Taker implied prices for any two-outcome SX market:
        {back1, back2, fair_one, overround}. None if the book is empty."""
        return self._devigged_fair(market)

    # ── odds math ────────────────────────────────────────────────────────────

    def _devigged_fair(self, market: dict) -> Optional[dict]:
        """Best-back implied per side from the order book, de-vigged. Fair for team one."""
        mh = market.get("marketHash")
        if not mh:
            return None
        cached = self._quote_cache.get(mh)
        if cached and (time.time() - cached[0]) < _QUOTE_TTL:
            return cached[1]
        data = self._get("/orders", {"marketHashes": mh})
        orders = (data or {}).get("data") or []
        # maker-implied odds by which outcome the maker is backing
        m1 = [int(o["percentageOdds"]) / 1e20 for o in orders
              if o.get("isMakerBettingOutcomeOne") and o.get("percentageOdds")]
        m2 = [int(o["percentageOdds"]) / 1e20 for o in orders
              if not o.get("isMakerBettingOutcomeOne") and o.get("percentageOdds")]
        # best price a taker gets to back an outcome = 1 - best maker-implied on the OTHER
        back1 = (1.0 - max(m2)) if m2 else None   # back team one
        back2 = (1.0 - max(m1)) if m1 else None   # back team two
        result = None
        if back1 and back2 and 0 < back1 < 1 and 0 < back2 < 1:
            fair1 = back1 / (back1 + back2)        # de-vig
            result = {"fair_one": fair1,
                      "back1": round(back1, 4), "back2": round(back2, 4),
                      "overround": round(back1 + back2 - 1.0, 4)}
        self._quote_cache[mh] = (time.time(), result)
        return result

    # ── public ───────────────────────────────────────────────────────────────

    def quote(self, player1: str, player2: str) -> Optional[dict]:
        """Live de-vigged fair probability that `player1` wins, plus score/meta.
        None if SX has no match-winner market for the fixture or the book is empty."""
        self._refresh_markets()
        s1, s2 = _surname(player1), _surname(player2)
        if not s1 or not s2:
            return None
        market = self._markets.get(frozenset((s1, s2)))
        if not market:
            return None
        fair = self._devigged_fair(market)
        if not fair:
            return None
        # orient fair to player1
        one_is_p1 = _surname(market.get("teamOneName", "")) == s1
        fair_p1 = fair["fair_one"] if one_is_p1 else (1.0 - fair["fair_one"])
        t1, t2 = market.get("teamOneScore"), market.get("teamTwoScore")
        score = None
        if t1 is not None and t2 is not None:
            score = f"{t1}-{t2}" if one_is_p1 else f"{t2}-{t1}"
        now = time.time()
        gt = market.get("gameTime") or 0
        # taker implied prices (what you'd pay to back each side), oriented to query
        back_p1 = fair["back1"] if one_is_p1 else fair["back2"]
        back_p2 = fair["back2"] if one_is_p1 else fair["back1"]
        return {
            "fair_p1": round(fair_p1, 4),
            "back_p1": round(back_p1, 4), "back_p2": round(back_p2, 4),
            "overround": fair["overround"],
            "score": score,
            "live": bool(gt) and now >= gt,
            "league": market.get("leagueLabel"),
            "market_hash": market.get("marketHash"),
        }

    def fair_prob(self, player1: str, player2: str) -> Optional[float]:
        q = self.quote(player1, player2)
        return q["fair_p1"] if q else None

    # ── execution (taker fill) — real money, guarded ──────────────────────────

    @property
    def can_trade_live(self) -> bool:
        return bool(os.getenv("SXBET_PRIVATE_KEY"))

    def _sign_fill(self, fill: dict) -> str:
        """EIP-712 sign a fill (Details wrapping FillObject) with the SX domain."""
        from eth_account import Account
        from eth_account.messages import encode_typed_data
        typed = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Details": [
                    {"name": "action", "type": "string"},
                    {"name": "market", "type": "string"},
                    {"name": "betting", "type": "string"},
                    {"name": "stake", "type": "string"},
                    {"name": "worstOdds", "type": "string"},
                    {"name": "worstReturning", "type": "string"},
                    {"name": "fills", "type": "FillObject"},
                ],
                "FillObject": [
                    {"name": "stakeWei", "type": "string"},
                    {"name": "marketHash", "type": "string"},
                    {"name": "baseToken", "type": "string"},
                    {"name": "desiredOdds", "type": "string"},
                    {"name": "oddsSlippage", "type": "uint256"},
                    {"name": "isTakerBettingOutcomeOne", "type": "bool"},
                    {"name": "fillSalt", "type": "uint256"},
                    {"name": "beneficiary", "type": "address"},
                    {"name": "beneficiaryType", "type": "uint8"},
                    {"name": "cashOutTarget", "type": "bytes32"},
                ],
            },
            "primaryType": "Details",
            "domain": {"name": "SX Bet", "version": DOMAIN_VERSION,
                       "chainId": self._meta_config()["chain_id"],
                       "verifyingContract": self._meta_config()["verifying_contract"]},
            "message": {
                "action": "N/A", "market": fill["marketHash"], "betting": "N/A",
                "stake": "N/A", "worstOdds": "N/A", "worstReturning": "N/A",
                "fills": fill,
            },
        }
        signable = encode_typed_data(full_message=typed)
        key = os.environ["SXBET_PRIVATE_KEY"]
        sig = Account.from_key(key).sign_message(signable).signature.hex()
        # eth_account >=0.13 returns bare hex; SX rejects a takerSig without the
        # 0x prefix ("must be a valid hex string of length 65 bytes").
        return sig if sig.startswith("0x") else "0x" + sig

    def place_bet(self, player1: str, player2: str, target_player: str,
                  stake_usd: float, odds_slippage: int = 3,
                  dry_run: bool = True) -> dict:
        """Place (or simulate) a taker fill backing `target_player` to win.

        dry_run builds + (if a key is present) signs the order but never posts.
        A live post requires SXBET_PRIVATE_KEY, a funded SX-Network wallet, and
        dry_run=False. Point SXBET_API at the testnet to rehearse for free.
        """
        self._refresh_markets()
        s_tgt = _surname(target_player)
        market = self._markets.get(frozenset((_surname(player1), _surname(player2))))
        if not market:
            return {"status": "no_market", "detail": "no SX match-winner market"}
        fair = self._devigged_fair(market)
        if not fair:
            return {"status": "no_book", "detail": "empty SX order book"}

        one_is_target = _surname(market.get("teamOneName", "")) == s_tgt
        label = f"back {target_player}"
        return self.place_fill(market, one_is_target, stake_usd, label,
                               odds_slippage=odds_slippage, dry_run=dry_run)

    def place_fill(self, market: dict, bet_outcome_one: bool, stake_usd: float,
                   label: str = "", odds_slippage: int = 3,
                   dry_run: bool = True) -> dict:
        """Place (or simulate) a taker fill on ANY two-outcome SX market —
        match winner, set winner, game spread, totals. Same guards as place_bet:
        dry_run builds + (with a key) signs but never posts."""
        fair = self._devigged_fair(market)
        if not fair:
            return {"status": "no_book", "detail": "empty SX order book"}
        implied = fair["back1"] if bet_outcome_one else fair["back2"]  # taker implied prob
        if not implied or not 0 < implied < 1:
            return {"status": "no_price", "detail": "no takeable price"}

        base_token = self._meta_config()["base_token"]
        fill = {
            "stakeWei": str(int(round(stake_usd * 1_000_000))),   # USDC 6dp
            "marketHash": market["marketHash"],
            "baseToken": base_token,
            "desiredOdds": str(int(round(implied * 10**20))),     # worst acceptable
            "oddsSlippage": int(odds_slippage),
            "isTakerBettingOutcomeOne": bool(bet_outcome_one),
            "fillSalt": str(int.from_bytes(secrets.token_bytes(32), "big")),
            "beneficiary": ZERO_ADDR, "beneficiaryType": 0,
            "cashOutTarget": ZERO_BYTES32,
        }
        live = (not dry_run) and self.can_trade_live
        outcome = market.get("outcomeOneName") if bet_outcome_one else market.get("outcomeTwoName")
        info = (f"{label or f'take {outcome}'} ${stake_usd:.2f} @ implied {implied:.3f} "
                f"(slip {odds_slippage}%) on {market.get('leagueLabel')}")
        if not live:
            sig = None
            try:                       # sign if a key exists, to prove the path
                if self.can_trade_live:
                    sig = self._sign_fill(fill)
            except Exception as e:
                return {"status": "sign_error", "detail": str(e)[:300]}
            return {"status": "dry_run", "detail": f"simulated fill: {info}",
                    "signed": bool(sig), "fill": fill, "implied": implied}
        try:
            taker = os.getenv("SXBET_WALLET_ADDRESS", "")
            sig = self._sign_fill(fill)
            # `market` must be the 32-byte market hash, NOT the "N/A" placeholder
            # used inside the signed Details struct.
            body = {"market": fill["marketHash"], "baseToken": base_token,
                    "isTakerBettingOutcomeOne": fill["isTakerBettingOutcomeOne"],
                    "stakeWei": fill["stakeWei"], "desiredOdds": fill["desiredOdds"],
                    "oddsSlippage": fill["oddsSlippage"], "fillSalt": fill["fillSalt"],
                    "taker": taker, "takerSig": sig}
            r = self.http.post(f"{API}/orders/fill/v2", json=body, timeout=20)
            ok = r.status_code == 200
            return {"status": "placed" if ok else "failed",
                    "detail": r.text[:300], "http": r.status_code, "implied": implied}
        except Exception as e:
            return {"status": "failed", "detail": str(e)[:300]}
