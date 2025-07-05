import pandas as pd
import numpy as np

exec(open('../Settings/combinations.sav').read())
exec(open('../CCRCs_clustering/Results/CCRC_dict_paper.sav').read())

combinations_selected=CCRCs_dict['selected_CCRC']
