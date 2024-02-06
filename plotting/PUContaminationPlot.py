import matplotlib.pyplot as plt
import mplhep as hep
from seaborn import regplot
from config.configuration import configuration

class PUContaminationPlot:
    def __init__(self, lc_dict_arr, annotate=None, save=None, legend=None, c=["red", "black"]):
        self.lc_dict_arr = lc_dict_arr
        
        self.annotate = annotate
        self.legend = legend
        self.c = c
        self.save = save

        self.makePlot()
        self.makeProfilePlot()

    def makePlot(self):
        fig1, ax1 = plt.subplots()
        for lc_dict, k in zip(self.lc_dict_arr, range(0, len(self.lc_dict_arr))):
            pu_contamination, lc_energy = self.getPUContamination(lc_dict)
            bin_content1, _, _ = ax1.hist(pu_contamination, histtype='step', linewidth=4, bins=20, label=self.legend[k], color=self.c[k])
        ax1.grid(True)
        ax1.set_xlabel("PU Contamination")
        ax1.legend()
        hep.cms.text(text=configuration["cmsText"], ax=ax1)
        ax1.annotate(self.annotate, xy=(0.1, max(bin_content1)*1.1))
                                                                                                                                               
        if (self.save != None):
            fig1.savefig(self.save+"pu_contamination.pdf")
        fig1.clf()
                                                                                                                                               
    def makeProfilePlot(self):
        fig1, ax1 = plt.subplots()
        for lc_dict, k in zip(self.lc_dict_arr, range(0, len(self.lc_dict_arr))):
            pu_contamination, lc_energy = self.getPUContamination(lc_dict)
            regplot(x=lc_energy, y=pu_contamination, x_bins=20, marker="o", color=self.c[k], fit_reg=None, label=self.legend[k], ax=ax1) 
                                                                                                                                               
        hep.cms.text(text=configuration["cmsText"], ax=ax1)
        ax1.set_xlabel("Energy [GeV]")
        ax1.set_ylabel("PU Contamination")
        ax1.legend()
        ax1.grid(True)
        if (self.save != None):
            fig1.savefig(self.save+"pu_contaminationProfile.pdf")
        
        fig1.clf()

    def getPUContamination(self, lc_dict):
        pu_contamination, lc_energy = [], []
        for event in range(0, len(lc_dict)):
            event = str(event+1)
            for lc in range(0, len(lc_dict[event]['PUContamination'])):
                already_matched = False
                for score in lc_dict[event]['LC2CPscore'][lc]:
                    if score < 0.2 and not already_matched:
                        pu_contamination.append(lc_dict[event]['PUContamination'][lc])
                        lc_energy.append(lc_dict[event]['LayerClustersEnergy'][lc])
                        already_matched = True
        return pu_contamination, lc_energy
                
