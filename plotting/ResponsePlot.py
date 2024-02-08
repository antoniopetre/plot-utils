import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
import numpy as np
from seaborn import regplot

from config.configuration import configuration
import config.style
from utils.computeProfile import computeProfile


class ResponsePlot:
    def __init__(self, cp_dict_arr, annotate=None, save=None, c=["red", "black"], legend=None):
        self.cp_dict_arr = cp_dict_arr
        
        self.annotate = annotate
        self.legend = legend
        self.c = c
        self.save = save

        self.bins = np.linspace(0, 2, 20)
        self.makePlot()
        self.makeProfilePlot()
    
    def makePlot(self):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        for cp_dict, k in zip(self.cp_dict_arr, range(0, len(self.cp_dict_arr))):
            response, cumulative_response, cp_energy = self.getSharedEnergy(cp_dict)
            bin_content1, _, _ = ax1.hist(response, histtype='step', linewidth=4, bins=self.bins, label=self.legend[k], color=self.c[k], density=True)
            bin_content2, _, _ = ax2.hist(cumulative_response, histtype='step', linewidth=4, bins=self.bins, label=self.legend[k], color=self.c[k], density=True)
        ax1.grid(True)
        ax2.grid(True)
        ax1.set_xlabel("Response")
        ax2.set_xlabel("Response")
        ax1.legend()
        ax2.legend()
        hep.cms.text(text=configuration["cmsText"], ax=ax1)
        hep.cms.text(text=configuration["cmsText"], ax=ax2)
        ax1.annotate(self.annotate, xy=(0.1, max(bin_content1)*1.1))
        ax2.annotate(self.annotate, xy=(0.1, max(bin_content2)*1.1))

        if (self.save != None):
            fig1.savefig(self.save+"response.pdf")
            fig2.savefig(self.save+"cumulative_response.pdf")
        fig1.clf()
        fig2.clf()

    def makeProfilePlot(self):
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        for cp_dict, k in zip(self.cp_dict_arr, range(0, len(self.cp_dict_arr))):
            response, cumulative_response, cp_energy = self.getSharedEnergy(cp_dict)
            regplot(x=cp_energy, y=response, x_bins=20, marker="o", color=self.c[k], fit_reg=None, label=self.legend[k], ax=ax1) 
            ax1.axhline(y=1, linestyle="--", linewidth=4, color="black")
            regplot(x=cp_energy, y=cumulative_response, x_bins=20, marker="o", color=self.c[k], fit_reg=None, label=self.legend[k], ax=ax2)   
            ax2.axhline(y=1, linestyle="--", linewidth=4, color="black")

        hep.cms.text(text=configuration["cmsText"], ax=ax1)
        hep.cms.text(text=configuration["cmsText"], ax=ax2)
        ax1.set_xlabel("Energy [GeV]")
        ax2.set_xlabel("Energy [GeV]")
        ax1.set_ylabel("Response")
        ax2.set_ylabel("Response")
        ax1.legend()
        ax2.legend()
        ax1.grid(True)
        ax2.grid(True)
        ax1.set_ylim(0.6, 1.4)
        ax2.set_ylim(0.6, 1.4)
        if (self.save != None):
            fig1.savefig(self.save+"responseProfile.pdf")
            fig2.savefig(self.save+"cumulative_responseProfile.pdf")
        
        fig1.clf()
        fig2.clf()

    def getSharedEnergy(self, cp_dict):
        response, cumulative_response, cp_energy = [], [], []
        for event in range(0, len(cp_dict)):
            event = str(event+1)
            for cp in range(0, len(cp_dict[event]['SharedEnergy'])):
                total_shared = 0
                if (len(cp_dict[event]['SharedEnergy'][cp])) == 0 : continue
                max_shared_energy = max(cp_dict[event]['SharedEnergy'][cp])
                max_shared_energy /= cp_dict[event]['CaloParticleEnergy'][cp]
                for shared in cp_dict[event]['SharedEnergy'][cp] :
                    total_shared += shared
                response.append(max_shared_energy)
                cumulative_response.append(total_shared / cp_dict[event]['CaloParticleEnergy'][cp])
                cp_energy.append(cp_dict[event]['CaloParticleEnergy'][cp])
        return response, cumulative_response, cp_energy
                    
