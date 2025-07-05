import pandas as pd
import numpy as np

if use_paper_model:
    reproduce_paper='_paper'
else:
    reproduce_paper=str()

exec(open('../Settings/combinations.sav').read())
exec(open('../CCRCs_clustering/Results/CCRC_dict'+reproduce_paper+'.sav').read())

combinations_selected=CCRC_dict['selected_CCRC']
