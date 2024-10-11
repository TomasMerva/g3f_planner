import pickle
import numpy as np
from texttable import Texttable
import latextable
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.record_data import RecordData


def table_results(file):
        # Save data
        results = pickle.load(open(file, 'rb'))
        n_runs = len(results)
        # --- create and plot table --- #
        rows = []
        title_row = [' ', "Success rate [\%]", 'Time-to-Success [s]', "Computation time[s]", "Collision-rate"]
        nr_column = len(title_row)
        rows.append(title_row)
        

        for case in results[0].keys():
             rows.append([
                    case,
                    str(np.round(np.sum([entry[case].goal_reached for entry in results]) / n_runs, decimals=1)) + " $\%$ ",
                    str(np.round(np.nanmean([entry[case].time_to_goal for entry in results]), decimals=4)) + " $\pm$ " + str(np.round(np.nanstd([entry[case].time_to_goal for entry in results]), decimals=4)),
                    str(np.round(np.nanmean(np.concatenate([entry[case].computation_time for entry in results], axis=0)),decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate([entry[case].computation_time for entry in results], axis=0)), decimals=6)),
                    str(np.round(np.sum([entry[case].collision for entry in results]) / n_runs, decimals=1))
             ])

        table = Texttable()
        table.set_cols_align(["c"] * nr_column)
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(rows)
        print('\nTexttable Latex:')
        print(latextable.draw_latex(table)) #, caption="\small{Statistics for 50 simulated scenarios of our proposed methods \ac{gm} and \ac{cm} compared to 50 scenarios of \ac{gf} and \ac{smp}}"))
      


if __name__=="__main__":
       result_file = "results/dinovas_tworobots_twotable.pickle"
       table_results(result_file)

