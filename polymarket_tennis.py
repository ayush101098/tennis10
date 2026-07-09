import json
import time
import requests
from typing import Optional

API_BASE_URL = "https://clob.polymarket.com"
KEYWORDS = ["tennis", "atp", "wta", "match", "set"]


class TennisMarketScanner:
    """Fetch and analyze tennis markets from Polymarket CLOB.

    This class is designed for junior developers to read live market data,
    compute implied probability, and estimate edge from an internal model.
    """

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def _get_json_with_retries(self, path: str, params: Optional[dict] = None):
        """Fetch JSON from the API with retry logic on HTTP errors.

        Retries up to 3 times with a 2 second backoff before raising the final exception.
        """
        url = f"{self.base_url}{path}"
        last_exception = None

        for attempt in range(1, 4):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, json.JSONDecodeError) as error:
                last_exception = error
                # Inform the developer if a retry is needed.
                print(
                    f"Attempt {attempt} failed for {url}: {error}. Retrying in 2 seconds..."
                )
                time.sleep(2)

        # After all attempts fail, raise the last seen exception.
        raise last_exception

    def _matches_tennis_keywords(self, question: str) -> bool:
        """Check if the market question contains any tennis-related keywords."""
        if not question:
            return False
        normalized = question.lower()
        return any(keyword in normalized for keyword in KEYWORDS)

    def _normalize_market(self, raw_market: dict) -> dict:
        """Normalize the API market response into a consistent dict format."""
        condition_id = raw_market.get("conditionId") or raw_market.get("condition_id")
        question = raw_market.get("question") or raw_market.get("description") or ""
        market_slug = raw_market.get("marketSlug") or raw_market.get("market_slug")
        yes_price = raw_market.get("yesPrice") or raw_market.get("yes_price") or raw_market.get("yes")
        no_price = raw_market.get("noPrice") or raw_market.get("no_price") or raw_market.get("no")
        volume = raw_market.get("volume") or raw_market.get("totalVolume") or 0.0
        end_date = raw_market.get("endDate") or raw_market.get("end_date")
        active = raw_market.get("active") if "active" in raw_market else raw_market.get("isActive", False)

        # Convert numeric values to Python types and handle missing data safely.
        try:
            yes_price = float(yes_price) if yes_price is not None else 0.0
        except (TypeError, ValueError):
            yes_price = 0.0

        try:
            no_price = float(no_price) if no_price is not None else 0.0
        except (TypeError, ValueError):
            no_price = 0.0

        try:
            volume = float(volume) if volume is not None else 0.0
        except (TypeError, ValueError):
            volume = 0.0

        return {
            "condition_id": condition_id,
            "question": question,
            "market_slug": market_slug,
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": volume,
            "end_date": end_date,
            "active": bool(active),
        }

    def fetch_active_markets(self) -> list[dict]:
        """Fetch all active tennis markets and return a filtered list of dictionaries."""
        raw_data = self._get_json_with_retries("/markets", params={"category": "sports"})

        # The API may return a list or a wrapper object with a key like "markets".
        raw_markets = raw_data if isinstance(raw_data, list) else raw_data.get("markets", [])
        tennis_markets = []

        for raw_market in raw_markets:
            normalized = self._normalize_market(raw_market)
            if normalized["active"] and self._matches_tennis_keywords(normalized["question"]):
                tennis_markets.append(normalized)

        return tennis_markets

    def get_implied_prob(self, condition_id: str) -> float:
        """Return the implied probability (yes_price) for a given market condition_id."""
        markets = self.fetch_active_markets()
        for market in markets:
            if market["condition_id"] == condition_id:
                return float(market["yes_price"])
        raise ValueError(f"Market with condition_id {condition_id} not found")

    def compute_edge(self, condition_id: str, true_p: float) -> dict:
        """Compute betting edge and Kelly fraction for a market using model true probability."""
        implied_prob = self.get_implied_prob(condition_id)
        edge = float(true_p) - implied_prob
        direction = "YES" if edge > 0 else "NO"

        # Avoid division by zero and invalid Kelly math when implied probability is 0 or 1.
        if implied_prob <= 0 or implied_prob >= 1:
            kelly_fraction = 0.0
        else:
            # quarter-Kelly formula as requested.
            numerator = true_p * (1 / implied_prob) - 1
            denominator = (1 / implied_prob) - 1
            if denominator == 0:
                kelly_fraction = 0.0
            else:
                kelly_fraction = 0.25 * (numerator / denominator)
                kelly_fraction = max(0.0, min(kelly_fraction, 0.05))

        bet_recommended = abs(edge) > 0.02
        return {
            "edge": edge,
            "direction": direction,
            "kelly_fraction": kelly_fraction,
            "bet_recommended": bet_recommended,
        }


def _truncate_text(value: str, max_length: int = 50) -> str:
    """Return a truncated version of the text for console display."""
    if not value:
        return ""
    return value if len(value) <= max_length else value[: max_length - 3] + "..."


def poll_markets(interval_seconds: int = 30) -> None:
    """Poll Polymarket tennis markets in a loop and print a console table every interval."""
    scanner = TennisMarketScanner()
    startup_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    try:
        tennis_markets = scanner.fetch_active_markets()
        status_line = f"Connected to {API_BASE_URL}."
        market_count = len(tennis_markets)
    except Exception as error:
        status_line = f"Failed to connect to {API_BASE_URL}: {error}"
        market_count = 0
        tennis_markets = []

    # Startup banner for developer visibility.
    print("\n=== Polymarket Tennis Scanner Startup ===")
    print(status_line)
    print(f"Tennis markets found: {market_count}")
    print(f"Timestamp: {startup_time}")
    print("======================================\n")

    while True:
        try:
            markets = scanner.fetch_active_markets()
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} - Latest tennis market data"
            )
            print(
                f"{'Question':50} | {'Implied':7} | {'TRUE P':6} | {'Edge':6} | {'Kelly':5}"
            )
            print("-" * 88)

            for market in markets:
                implied_prob = market["yes_price"]
                true_p = 0.5
                edge = true_p - implied_prob
                kelly = 0.0
                if implied_prob > 0 and implied_prob < 1:
                    numerator = true_p * (1 / implied_prob) - 1
                    denominator = (1 / implied_prob) - 1
                    if denominator != 0:
                        kelly = 0.25 * (numerator / denominator)
                        kelly = max(0.0, min(kelly, 0.05))

                print(
                    f"{_truncate_text(market['question'], 50):50} | "
                    f"{implied_prob:7.3f} | {true_p:6.3f} | {edge:6.3f} | {kelly:5.3f}"
                )

            print("\n")
        except Exception as error:
            print(f"Error polling markets: {error}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    poll_markets()
