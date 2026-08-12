"use client";

import Link from "next/link";
import Socials from "@/components/Socials";
import { DonateLink } from "@/components/Donate";
import { LEGAL_NAME } from "@/lib/brand";
import { planById } from "@/lib/plans";

/**
 * Site footer.
 *
 * The risk disclosure is deliberately IN the footer and not buried in a legal
 * page: this product tells people how much money to put on a tennis match, and
 * the one line everyone sees on every page should be the one that says the
 * model can be wrong.
 */
export default function SiteFooter() {
  const year = new Date().getFullYear();

  return (
    <footer className="border-t border-terminal-border mt-4">
      <div className="max-w-[1000px] mx-auto px-6 py-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-[11px]">
          <div className="col-span-2 md:col-span-1">
            <div className="text-terminal-green font-bold mb-2">◉ Tennis Alpha</div>
            <p className="text-terminal-muted leading-relaxed">
              Live win probability, market edge and Kelly-disciplined staking for professional
              tennis — ATP, WTA, Challenger, W125 and ITF.
            </p>
            <div className="mt-3">
              <Socials variant="footer" />
            </div>
          </div>

          <nav aria-label="Product">
            <div className="text-slate-200 font-bold mb-2">Product</div>
            <ul className="space-y-1.5 text-terminal-muted">
              <li><Link href="/terminal" className="hover:text-slate-200">Terminal</Link></li>
              <li><Link href="/#matches" className="hover:text-slate-200">Today&apos;s matches</Link></li>
              <li><Link href="/calculator" className="hover:text-slate-200">Kelly calculator</Link></li>
              <li><Link href="/#faq" className="hover:text-slate-200">FAQ</Link></li>
            </ul>
          </nav>

          <nav aria-label="Learn">
            <div className="text-slate-200 font-bold mb-2">Learn</div>
            <ul className="space-y-1.5 text-terminal-muted">
              <li><Link href="/manual" className="hover:text-slate-200">Trading manual</Link></li>
              <li><Link href="/#manual" className="hover:text-slate-200">Video walkthrough</Link></li>
              <li><Link href="/calculator" className="hover:text-slate-200">Bankroll growth</Link></li>
            </ul>
          </nav>

          <nav aria-label="Company">
            <div className="text-slate-200 font-bold mb-2">Company</div>
            <ul className="space-y-1.5 text-terminal-muted">
              <li>
                <a href="mailto:jessefuture10@gmail.com" className="hover:text-slate-200">Contact</a>
              </li>
              <li>
                <span className="text-terminal-muted">From ${planById("day").usd}/day</span>
              </li>
              <li><DonateLink /></li>
            </ul>
          </nav>
        </div>

        <div className="mt-8 pt-5 border-t border-terminal-border text-[9px] text-terminal-muted leading-relaxed">
          <p>
            <b className="text-slate-400">Risk.</b> Model outputs are calibrated probabilities, not
            guarantees, and the model can be wrong. Sports betting involves risk — bet only what you
            can afford to lose. Staking discipline (¼ Kelly, 5% bankroll cap, 2% edge floor) is
            enforced in the product for a reason. Tennis Alpha is an analytics tool, not a bookmaker
            and not betting advice; whether betting is legal where you live is your responsibility.
          </p>
          <p className="mt-2">© {year} {LEGAL_NAME}. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}
