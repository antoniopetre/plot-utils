import mplhep as hep
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time

from data.dictionaries import fillLCDictionary, fillCPDictionary, fillSCDictionary
from plotting.EfficiencyPlots import EfficiencyPlots
from plotting.PurityPlots import PurityPlots
from plotting.FakeRatePlots import FakeRatePlots
from plotting.ResponsePlot import ResponsePlot
from plotting.PUContaminationPlot import PUContaminationPlot
from plotting.ClusterPlots import ClusterPlots

from config.configuration import configuration


if __name__ == "__main__":

    dir_string = configuration["FilesDirectory"] 
    files_vec = []
    files = list(os.listdir(dir_string))
    files = [dir_string+f for f in files]
    files_vec.extend(files)
   
    if (configuration["DebugMode"]):
        print(">>> Running in DebugMode")
        files_vec = files_vec[:10] 
    print(">>> Found %s files" %len(files_vec))

    doPFComparison = configuration["doPFComparison"]
    
    start = time.time()
    lc_dict = fillLCDictionary(files_vec)
    cp_dict = fillCPDictionary(files_vec)
    sc_dict = fillSCDictionary(files_vec)

    if (doPFComparison):
        lcpf_dict = fillLCDictionary(files_vec, isPF=True)
        cppf_dict = fillCPDictionary(files_vec, isPF=True)
        scpf_dict = fillSCDictionary(files_vec, isPF=True)

    stop = time.time()

    print(">>> Dictionaries ready; elapsed time: {:.2f} seconds".format(stop-start))

    ClusterPlots([lc_dict, lcpf_dict], legend=["LayerCluster", "PFClusters"], c=["red", "black"], save=configuration["SaveDirectory"])
    
    if (doPFComparison):
        EfficiencyPlots([lc_dict, lcpf_dict], [cp_dict, cppf_dict],\
                        legend=["CLUE", "PFClustering"],
                        annotate=configuration["Annotate"],
                        save=configuration["SaveDirectory"])
        PurityPlots([lc_dict, lcpf_dict], [cp_dict, cppf_dict],\
                    legend=["CLUE", "PFClustering"],
                    annotate=configuration["Annotate"],
                    save=configuration["SaveDirectory"])
        FakeRatePlots([lc_dict, lcpf_dict], [cp_dict, cppf_dict],\
                      legend=["CLUE", "PFClustering"],
                      annotate=configuration["Annotate"], 
                      save=configuration["SaveDirectory"])
    else :
        EfficiencyPlots([lc_dict], [cp_dict], c=['red'], legend=["CLUE"], save=configuration["SaveDirectory"])
        PurityPlots([lc_dict], [cp_dict], c=['red'], legend=["CLUE"], save=configuration["SaveDirectory"])
        FakeRatePlots([lc_dict], [cp_dict], c=['red'], legend=["CLUE"], save=configuration["SaveDirectory"])

    # make specific plots
    ResponsePlot([cp_dict, cppf_dict], 
                 annotate=configuration["Annotate"],
                 legend=["CLUE", "PFClustering"],
                 c=['red', 'black'],
                 save=configuration["SaveDirectory"])

    PUContaminationPlot([lc_dict], 
                        annotate=configuration["Annotate"],
                        legend=["CLUE"],
                        c=['red'],
                        save=configuration["SaveDirectory"])



    print(">>> Plots saved in %s" % configuration["SaveDirectory"])

