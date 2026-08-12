import type { Metadata } from "next";
import Link from "next/link";
import Wordmark from "@/components/Wordmark";
import Socials from "@/components/Socials";
import SiteFooter from "@/components/SiteFooter";
import { BreadcrumbLd } from "@/components/JsonLd";

export const metadata: Metadata = {
  title: "Tennis Modelling Research — Papers on Prediction, Elo & Betting Markets | Tennis Alpha",
  description:
    "A reading list of peer-reviewed research behind tennis win-probability modelling: Markov point models, Elo variants, Bayesian hierarchical models, machine learning and betting-market efficiency.",
  alternates: { canonical: "/resources" },
  openGraph: {
    title: "Tennis Modelling Research — the papers behind the model",
    description: "Peer-reviewed work on tennis prediction, Elo, Bayesian models and market efficiency.",
    url: "/resources",
  },
};

/**
 * Research reading list.
 *
 * Every entry was checked against a live search result before being listed —
 * no citation here was written from memory. Where a paywalled paper has a free
 * author copy or repository PDF, that link is given as well, because a
 * reference nobody can open is not much of a reference.
 *
 * The findings are summarised honestly, including the ones that cut against
 * selling a betting product: several of these papers conclude that models
 * struggle to beat the closing price. A research page that hid that would be
 * worth less than no research page.
 */

interface Paper {
  authors: string;
  year: string;
  title: string;
  venue: string;
  url: string;
  free?: string;
  note: string;
}

const SECTIONS: { heading: string; blurb: string; papers: Paper[] }[] = [
  {
    heading: "Point-level and Markov models",
    blurb:
      "The foundation the live engine rests on: a match as a sequence of service points, re-priced from the score.",
    papers: [
      {
        authors: "Klaassen & Magnus", year: "2003",
        title: "Forecasting the winner of a tennis match",
        venue: "European Journal of Operational Research 148(2), 257–267",
        url: "https://www.sciencedirect.com/science/article/abs/pii/S0377221702006823",
        free: "https://www.janmagnus.nl/papers/JRM065.pdf",
        note: "The canonical in-play forecasting paper — win probability updated point by point, not just before the match.",
      },
      {
        authors: "Barnett & Clarke", year: "2005",
        title: "Combining player statistics to predict outcomes of tennis matches",
        venue: "IMA Journal of Management Mathematics 16(2), 113–120",
        url: "https://academic.oup.com/imaman/article-abstract/16/2/113/704903",
        note: "How to turn published serve statistics into the point-win probabilities a Markov model needs.",
      },
      {
        authors: "Knottenbelt, Spanias & Madurska", year: "2012",
        title: "A common-opponent stochastic model for predicting the outcome of professional tennis matches",
        venue: "Computers & Mathematics with Applications 64(12), 3820–3827",
        url: "https://www.sciencedirect.com/science/article/pii/S0898122112002106",
        note: "Compares two players through opponents they have both faced, rather than through raw averages.",
      },
      {
        authors: "Ingram", year: "2019",
        title: "A point-based Bayesian hierarchical model to predict the outcome of tennis matches",
        venue: "Journal of Quantitative Analysis in Sports 15(4), 313–325",
        url: "https://www.degruyterbrill.com/document/doi/10.1515/jqas-2018-0008/html",
        free: "https://martiningram.github.io/papers/bayes_point_based.pdf",
        note: "Serve and return skill as a Gaussian random walk, varying by surface. Reports 68.8% accuracy against 66.3% for earlier point-based models.",
      },
      {
        authors: "University of Glasgow (MSci)", year: "—",
        title: "Predicting the outcome of tennis matches from point-by-point data",
        venue: "MSci project, School of Computing Science",
        url: "https://www.dcs.gla.ac.uk/~srogers/files/projects/MSci_project_1006404b.pdf",
        note: "A readable, self-contained walk through point-level prediction if you want the mechanics before the algebra.",
      },
    ],
  },
  {
    heading: "Ratings systems",
    blurb: "Elo and its descendants — still the benchmark any new model has to beat.",
    papers: [
      {
        authors: "Kovalchik", year: "2016",
        title: "Searching for the GOAT of tennis win prediction",
        venue: "Journal of Quantitative Analysis in Sports 12(3), 127–138",
        url: "https://www.degruyter.com/document/doi/10.1515/jqas-2015-0059/html",
        free: "https://vuir.vu.edu.au/34652/1/jqas-2015-0059.pdf",
        note: "Eleven published models tested head to head on 2,395 ATP matches. The single most useful benchmark paper in the field.",
      },
      {
        authors: "Angelini, Candila & De Angelis", year: "2022",
        title: "Weighted Elo rating for tennis match predictions",
        venue: "European Journal of Operational Research 297(1), 120–132",
        url: "https://www.sciencedirect.com/science/article/abs/pii/S0377221721003234",
        free: "https://cris.unibo.it/bitstream/11585/821483/2/Weighted%20ELO%20rating%20predictions%20in%20tennis.pdf",
        note: "Weights each result by how emphatic it was, rather than treating every win as identical.",
      },
      {
        authors: "Kovalchik", year: "2020",
        title: "Extension of the Elo rating system to margin of victory",
        venue: "International Journal of Forecasting 36(4), 1329–1341",
        url: "https://www.sciencedirect.com/science/article/abs/pii/S0169207020300157",
        note: "Four ways to fold margin of victory into Elo. Only the joint additive form stayed unbiased in simulation.",
      },
      {
        authors: "PLOS ONE", year: "2022",
        title: "A study of forecasting tennis matches via the Glicko model",
        venue: "PLOS ONE 17(4)",
        url: "https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0266838",
        note: "Glicko adds a reliability term to Elo — useful for players with thin recent histories, which is most of the ITF draw.",
      },
      {
        authors: "Scientific Reports", year: "2022",
        title: "A new model for predicting the winner in tennis based on eigenvector centrality",
        venue: "Open access via PMC",
        url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC8900648/",
        note: "Treats the tour as a network of results and ranks players by their position in it.",
      },
    ],
  },
  {
    heading: "Machine learning",
    blurb: "What modern learners add over a well-specified statistical model — and what they do not.",
    papers: [
      {
        authors: "Wilkens", year: "2021",
        title: "Sports prediction and betting models in the machine learning age: the case of tennis",
        venue: "Journal of Sports Analytics 7(2), 99–117",
        url: "https://doi.org/10.3233/JSA-200463",
        free: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3506302",
        note: "Read this one before believing any ML claim: the models beat ranking-only baselines but could not beat odds-implied forecasts.",
      },
      {
        authors: "Buhamra, Groll & Brunner", year: "2024",
        title: "Modeling and prediction of tennis matches at Grand Slam tournaments",
        venue: "Journal of Sports Analytics",
        url: "https://journals.sagepub.com/doi/10.3233/JSA-240670",
        note: "Recent, and specific about Grand Slams, where best-of-five changes the variance structure.",
      },
      {
        authors: "Groll et al.", year: "2025",
        title: "Statistical enhanced learning for modeling and prediction of tennis matches at Grand Slam tournaments",
        venue: "arXiv:2502.01613",
        url: "https://arxiv.org/abs/2502.01613",
        note: "Feeds statistical model output into a learner rather than choosing between the two.",
      },
      {
        authors: "Cornman, Spellman & Wright", year: "2017",
        title: "Machine learning for professional tennis match prediction and betting",
        venue: "Stanford CS229 project report",
        url: "https://cs229.stanford.edu/proj2017/final-reports/5242116.pdf",
        note: "Short and practical, and unusually candid about how thin the betting margin turned out to be.",
      },
      {
        authors: "Lerner", year: "2019",
        title: "DeepTennis: mid-match tennis predictions",
        venue: "Stanford CS230 project report",
        url: "https://cs230.stanford.edu/projects_fall_2019/reports/26249098.pdf",
        note: "In-play prediction with neural networks — the closest academic analogue to the live re-pricing here.",
      },
      {
        authors: "Springer (MLSA)", year: "2021",
        title: "Predicting tennis match outcomes with network analysis and machine learning",
        venue: "Machine Learning and Data Mining for Sports Analytics",
        url: "https://link.springer.com/chapter/10.1007/978-3-030-67731-2_37",
        note: "Network features as inputs to a learner rather than as a ranking in their own right.",
      },
      {
        authors: "arXiv", year: "2024",
        title: "Capturing momentum: tennis match analysis using machine learning and time series theory",
        venue: "arXiv:2404.13300",
        url: "https://arxiv.org/abs/2404.13300",
        note: "Tests whether momentum is measurable rather than assuming it. Relevant to any in-play signal.",
      },
    ],
  },
  {
    heading: "Betting markets and efficiency",
    blurb:
      "The part most model write-ups skip: whether an edge survives contact with the price you can actually get.",
    papers: [
      {
        authors: "Easton & Uylangco", year: "2010",
        title: "Forecasting outcomes in tennis matches using within-match betting markets",
        venue: "International Journal of Forecasting 26(3), 544–553",
        url: "https://www.sciencedirect.com/science/article/abs/pii/S0169207009001721",
        note: "In-play odds track the match closely — the benchmark any live model is competing against.",
      },
      {
        authors: "Forrest & McHale", year: "2007",
        title: "Longshot bias: insights from the betting market on men's professional tennis",
        venue: "Information Efficiency in Financial and Betting Markets (Cambridge UP), ch. 8",
        url: "https://www.cambridge.org/core/books/abs/information-efficiency-in-financial-and-betting-markets/longshot-bias-insights-from-the-betting-market-on-mens-professional-tennis/C35909D718F8D842B81F85FADFF0A735",
        note: "Finds longshot bias in tennis, but not enough of it to make backing favourites profitable.",
      },
      {
        authors: "Ramirez, Reade & Singleton", year: "2023",
        title: "Betting on a buzz: mispricing and inefficiency in online sportsbooks",
        venue: "International Journal of Forecasting 39(3)",
        url: "https://www.sciencedirect.com/science/article/pii/S0169207022001091",
        free: "https://www.reading.ac.uk/web/files/economics/emdp202110.pdf",
        note: "Where sportsbook prices drift from fair value, using tennis as the setting.",
      },
      {
        authors: "arXiv", year: "2024",
        title: "A systematic review of machine learning in sports betting: techniques, challenges and future directions",
        venue: "arXiv:2410.21484",
        url: "https://arxiv.org/abs/2410.21484",
        note: "A survey worth reading for the recurring methodological errors it catalogues.",
      },
      {
        authors: "Springer", year: "2018",
        title: "Predicting the outcome of a tennis tournament: based on both data and judgments",
        venue: "Journal of Systems Science and Systems Engineering 27",
        url: "https://link.springer.com/content/pdf/10.1007%2Fs11518-018-5395-3.pdf",
        note: "Combining model output with human judgement, and when that helps rather than hurts.",
      },
    ],
  },
];

const TOTAL = SECTIONS.reduce((n, s) => n + s.papers.length, 0);

export default function ResourcesPage() {
  return (
    <div className="marketing min-h-screen bg-terminal-bg text-slate-200">
      <BreadcrumbLd trail={[{ name: "Research", path: "/resources" }]} />

      <nav className="sticky top-0 z-40 flex items-center justify-between gap-2 px-3 sm:px-6 py-3 border-b border-terminal-border bg-terminal-bg/95 backdrop-blur">
        <Link href="/" className="hover:opacity-80"><Wordmark size={16} /></Link>
        <div className="flex items-center gap-2 sm:gap-3 text-[11px] shrink-0">
          <Socials />
          <Link href="/manual" className="hidden sm:inline text-terminal-muted hover:text-slate-200">Manual</Link>
          <Link href="/terminal"
            className="inline-flex items-center justify-center min-h-[40px] font-bold px-3 rounded bg-terminal-green text-black hover:opacity-90">
            OPEN TERMINAL →
          </Link>
        </div>
      </nav>

      <main className="px-4 sm:px-6 py-10 max-w-[900px] mx-auto">
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-100 mb-3">The research behind the model</h1>
        <p className="text-[13px] text-slate-400 leading-relaxed max-w-[640px] mb-2">
          {TOTAL} peer-reviewed papers and academic reports on tennis win-probability modelling — the
          Markov point models, rating systems, learners and market studies this terminal is built on
          and measured against.
        </p>
        <p className="text-[11px] text-terminal-muted leading-relaxed max-w-[640px] mb-8">
          Several of these conclude that models struggle to beat the closing price. They are listed
          anyway, and summarised as they read — a reading list that only cited the encouraging half
          would tell you nothing about the field.
        </p>

        {SECTIONS.map(sec => (
          <section key={sec.heading} className="mb-10">
            <h2 className="text-[15px] font-bold text-slate-100 mb-1">{sec.heading}</h2>
            <p className="text-[11px] text-terminal-muted mb-4">{sec.blurb}</p>
            <ol className="border border-terminal-border rounded-lg overflow-hidden">
              {sec.papers.map(p => (
                <li key={p.title} className="border-b border-terminal-border last:border-b-0 p-4 hover:bg-terminal-panel/30 transition">
                  <a href={p.url} target="_blank" rel="noreferrer"
                    className="text-[13px] font-bold text-slate-100 hover:text-terminal-green">
                    {p.title} ↗
                  </a>
                  <div className="text-[11px] text-terminal-muted mt-1">
                    {p.authors} · {p.year} · <span className="italic">{p.venue}</span>
                  </div>
                  <p className="text-[11.5px] text-slate-400 mt-2 leading-relaxed">{p.note}</p>
                  {p.free && (
                    <a href={p.free} target="_blank" rel="noreferrer"
                      className="inline-block mt-2 text-[10px] text-terminal-cyan hover:underline">
                      free PDF ↗
                    </a>
                  )}
                </li>
              ))}
            </ol>
          </section>
        ))}

        <div className="border border-terminal-green/30 bg-terminal-green/[0.06] rounded-lg p-4 text-center">
          <p className="text-[12px] text-slate-200 mb-3">
            The terminal is this literature, running live on today&apos;s matches.
          </p>
          <Link href="/terminal"
            className="inline-flex items-center justify-center min-h-[44px] px-5 rounded bg-terminal-green text-black text-xs font-bold hover:opacity-90">
            OPEN THE TERMINAL →
          </Link>
        </div>
      </main>

      <SiteFooter />
    </div>
  );
}
