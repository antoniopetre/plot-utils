import matplotlib.pyplot as plt
import mplhep as hep
from config.configuration import configuration

class ClusterPlots:
    def __init__(self, lc_dict_arr, legend=None, c=["red", "black"], annotate=None, save=None):
        self.lc_dict_arr = lc_dict_arr
        
        self.legend = legend
        self.c = c
        self.annotate = annotate
        self.save = save
        self.makePlot()

    def makePlot(self):
        self.nClustersPlot()
        self.clustersEta()
        self.clustersPhi()
        self.clustersEnergy()
        self.clustersLayer()

    def nClustersPlot(self):
        binContent = []
        for lc_dict, k in zip(self.lc_dict_arr, range(0, len(self.lc_dict_arr))):
            nClustersPerEvent = []
            for event in range(0, len(lc_dict)):
                event = str(event+1)
                nClustersPerEvent.append(len(lc_dict[event]['LayerClustersEnergy']))
            bin_content, _, _ = plt.hist(nClustersPerEvent, 
                                         histtype="step", color=self.c[k], linewidth=4, label=self.legend[k], bins=50)
            binContent.append(max(bin_content))
        plt.grid(True)
        plt.xlabel("Number of clusters per event")
        hep.cms.text(text=configuration["cmsText"])
        plt.legend()
        if (self.annotate != None):
            plt.annotate(text=configuration["Annotate"], xy=(70, max(bin_content)))
        
        if (self.save != None):
            plt.savefig(self.save+"nClustersPerEvent.pdf")

        plt.clf()

    def clustersEta(self):
        binContent = []
        for lc_dict, k in zip(self.lc_dict_arr, range(0, len(self.lc_dict_arr))):
            eta = []
            for event in range(0, len(lc_dict)):
                event = str(event+1)
                for lc in lc_dict[event]['LayerClustersEta']:
                    eta.append(lc)
            bin_content, _, _ = plt.hist(eta,
                                         histtype="step", color=self.c[k], linewidth=4, label=self.legend[k], bins=20)
            binContent.append(max(bin_content))
        plt.grid(True)
        plt.xlabel(r"$\eta$")
        hep.cms.text(text=configuration["cmsText"])
        plt.legend()
        
        if (self.save != None):
            plt.savefig(self.save+"clustersEta.pdf")

        plt.clf()

    def clustersPhi(self):
        binContent = []
        for lc_dict, k in zip(self.lc_dict_arr, range(0, len(self.lc_dict_arr))):
            phi = []
            for event in range(0, len(lc_dict)):
                event = str(event+1)
                for lc in lc_dict[event]['LayerClustersPhi']:
                    phi.append(lc)
            bin_content, _, _ = plt.hist(phi,
                                         histtype="step", color=self.c[k], linewidth=4, label=self.legend[k], bins=20)
            binContent.append(max(bin_content))
        plt.grid(True)
        plt.xlabel(r"$\phi$")
        hep.cms.text(text=configuration["cmsText"])
        plt.legend()
        
        if (self.save != None):
            plt.savefig(self.save+"clustersPhi.pdf")

        plt.clf()

    def clustersEnergy(self):
        binContent = []
        for lc_dict, k in zip(self.lc_dict_arr, range(0, len(self.lc_dict_arr))):
            ene = []
            for event in range(0, len(lc_dict)):
                event = str(event+1)
                for lc in lc_dict[event]['LayerClustersEnergy']:
                    ene.append(lc)
            bin_content, _, _ = plt.hist(ene,
                                         histtype="step", color=self.c[k], linewidth=4, label=self.legend[k], bins=20)
            binContent.append(max(bin_content))
        plt.grid(True)
        plt.xlabel("Energy [GeV]")
        plt.yscale('log')
        hep.cms.text(text=configuration["cmsText"])
        plt.legend()
        
        if (self.save != None):
            plt.savefig(self.save+"clustersEnergy.pdf")

        plt.clf()

    def clustersLayer(self):
        binContent = []
        for lc_dict, k in zip(self.lc_dict_arr, range(0, len(self.lc_dict_arr))):
            layer = []
            for event in range(0, len(lc_dict)):
                event = str(event+1)
                for lc in lc_dict[event]['LayerClustersLayer']:
                    layer.append(lc)
            bin_content, _, _ = plt.hist(layer,
                                         histtype="step", color=self.c[k], linewidth=4, label=self.legend[k], bins=list(range(0, 7,1)))
            binContent.append(max(bin_content))
        plt.grid(True)
        plt.xlabel("Layer")
        hep.cms.text(text=configuration["cmsText"])                                                                      
        plt.legend()
        
        if (self.save != None):
            plt.savefig(self.save+"clustersLayer.pdf")

        plt.clf()
