SUPPLY CHAIN DISRUPTION & FINANCIAL LOSS SIMULATOR
Student: M. Abdullah Haroon | BSE-23-10 | Stochastic Calculus
7th Semester, Evening, Group A | 22 November 2026
===========================================================

FOLDER CONTENTS
  supply-chain-simulator.html   Web app. Double-click to open in any browser
                                (phone or computer, works offline). Tutorial inside.
  sc_simulator.py               Python simulator (full analysis, plots, CSV)
  report/
    Supply_Chain_Simulator_Report_4pages_BW.pdf   4-page report, black-and-white print version (submit this one)
    Supply_Chain_Simulator_Report_4pages.pdf      4-page report, colour version
    Supply_Chain_Simulator_Report.pdf             Full 8-page report
  figures/color, figures/black_white   Plots and policy_summary.csv from the 40,000-path run

HOW TO RUN THE PYTHON SIMULATOR
  pip install numpy scipy matplotlib
  python sc_simulator.py --paths 40000 --out results

NOTE
  All parameters are examples. Replace them with real data before making real decisions.
