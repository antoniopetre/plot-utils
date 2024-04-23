import matplotlib.pyplot as plt
import mplhep as hep
from config.configuration import configuration
import awkward as ak
import numpy as np

class ClusterPlots:
    def __init__(self, lc_dict_arr, lcpf_dict, legend=None, c=["red", "black"], annotate=None,
                        save=None, debug=None, recHits=None):
        self.lc_dict_arr = lc_dict_arr
        self.lcpf_dict_arr = lcpf_dict
        self.legend = legend
        self.c = c
        self.annotate = annotate
        self.save = save
        self.debug = debug
        self.recHits = recHits
        self.makePlot()
        
    def makePlot(self):
        # 1st plot nClustersPerEvent
        nClustersPerEvent_lc = ak.count(self.lc_dict_arr['layerClusterEnergy'], axis=1)
        nClustersPerEvent_lcpf = ak.count(self.lcpf_dict_arr['layerClusterEnergy'], axis=1)
        

        binsEta, _ = np.linspace(start=-1.5, stop=1.5, retstep=0.0175)
        binsPhi, _ = np.linspace(start=-np.pi, stop=np.pi, retstep=0.0175)

        clustersEta = ak.flatten(self.lc_dict_arr['layerClusterEta']).to_numpy()
        clustersPhi = ak.flatten(self.lc_dict_arr['layerClusterPhi']).to_numpy()
        self.hist_2d(clustersEta, clustersPhi, xlabel='LC ETA', ylabel='LC PHI', title='',
                pdf_name='abc.png', logY=False, binning=(binsEta, binsPhi))

        mask = nClustersPerEvent_lc == 0
        mask2 = nClustersPerEvent_lc > 0
        caloEta_masked = ak.flatten(self.debug['caloParticleEta'][mask]).to_numpy()
        caloPhi_masked = ak.flatten(self.debug['caloParticlePhi'][mask]).to_numpy()
        self.hist_2d(caloEta_masked, caloPhi_masked, xlabel='CaloP ETA', ylabel='CaloP PHI', title='CaloP when #LC = 0',
                pdf_name='eta_phi_masked.png', logY=False, binning=(binsEta, binsPhi))


        recHitsEta = ak.flatten(self.recHits['pfrechitEta']).to_numpy()
        recHitsPhi = ak.flatten(self.recHits['pfrechitPhi']).to_numpy()
        self.hist_2d(recHitsEta, recHitsPhi, xlabel='RecHits ETA', ylabel='RecHits PHI', title='',
                pdf_name='eta_phi_recHits.png', logY=False, binning=(binsEta, binsPhi))

        recHitsEta_masked = ak.flatten(self.recHits['pfrechitEta'][mask]).to_numpy()
        recHitsPhi_masked = ak.flatten(self.recHits['pfrechitPhi'][mask]).to_numpy()
        self.hist_2d(recHitsEta_masked, recHitsPhi_masked, xlabel='RecHits ETA', ylabel='RecHits PHI', title='RecHits when #LC = 0',
                pdf_name='eta_phi_recHits_masked.png', logY=False, binning=(binsEta, binsPhi))
        
        self.plot_hist1d_debug(ak.flatten(self.debug['caloParticleEnergy'][mask]), xlabel='Energy', pdf_name='caloParticlesEnergy_masked.png')
        self.plot_hist1d_debug(ak.flatten(self.debug['caloParticleEta'][mask]), xlabel=r'\eta', pdf_name='caloParticlesEta_masked.png')
        self.plot_hist1d_debug(ak.flatten(self.debug['caloParticlePhi'][mask]), xlabel=r'\phi', pdf_name='caloParticlesPhi_masked.png')

        x = self.debug['caloParticleEnergy'][mask]
        y = self.debug['caloParticleEta'][mask]

        No_total_ev = len(self.lc_dict_arr['layerClusterNumberOfHits'])

        En_cut = 50
        mask_new0 = x > En_cut
        no_events0 = ak.count_nonzero(mask_new0)
        string0 = f'1st cut: # LC/event = 0 \n 2nd cut: En of CaloP > {En_cut} \n {no_events0}/{No_total_ev} events'
        self.plot_hist1d_debug(ak.flatten(x[mask_new0]), xlabel='CaloParticle Energy',
                            pdf_name='caloParticlesEnergy_masked1.png', annotate=string0)
        self.plot_hist1d_debug(ak.flatten(y[mask_new0]), xlabel=r'CaloParticle \eta',
                            pdf_name='caloParticlesEta_masked1.png', annotate=string0)

        eta_cut = 0.2
        mask_new1 = np.abs(ak.flatten(y).to_numpy()) > eta_cut
        no_events1 = np.count_nonzero(mask_new1)
        string1 = f'1st cut: # LC/event = 0 \n 2nd cut: abs(eta) of CaloP > {eta_cut} \n {no_events1}/{No_total_ev} events'
        self.plot_hist1d_debug(ak.flatten(x)[mask_new1], xlabel='CaloParticle Energy',
                            pdf_name='caloParticlesEnergy_masked2.png', annotate=string1)
        self.plot_hist1d_debug(ak.flatten(y)[mask_new1], xlabel=r'CaloParticle \eta',
                            pdf_name='caloParticlesEta_masked2.png', annotate=string1)
        self.plot_hist1d(nClustersPerEvent_lc, nClustersPerEvent_lcpf,
                        xlabel='Number of clusters per event', pdf_name='nClustersPerEvent.pdf', binning=50)

        # 2nd plot clustersEta
        self.plot_hist1d(ak.flatten(self.lc_dict_arr['layerClusterEta']), ak.flatten(self.lcpf_dict_arr['layerClusterEta']),
                        xlabel=r"$\eta$", pdf_name='clustersEta.pdf', binning=20)

        # 3rd plot clustersPhi
        self.plot_hist1d(ak.flatten(self.lc_dict_arr['layerClusterPhi']), ak.flatten(self.lcpf_dict_arr['layerClusterPhi']),
                        xlabel=r"$\phi$", pdf_name='clustersPhi.pdf', binning=20)

        # 4th plot clustersEnergy
        self.plot_hist1d(ak.flatten(self.lc_dict_arr['layerClusterEnergy']), ak.flatten(self.lcpf_dict_arr['layerClusterEnergy']),
                        xlabel='Energy [GeV]', pdf_name='clustersEnergy.pdf', logY=True, binning=20)

        # 5th plot clustersLayer
        self.plot_hist1d(ak.flatten(self.lc_dict_arr['layerClusterLayer']), ak.flatten(self.lcpf_dict_arr['layerClusterLayer']),
                        xlabel='Layer', pdf_name='clustersLayer.pdf', binning=list(range(0,7,1)))

        # 6th plot nHitsPerCluster
        self.plot_hist1d(ak.flatten(self.lc_dict_arr['layerClusterNumberOfHits']), ak.flatten(self.lcpf_dict_arr['layerClusterNumberOfHits']),
                        xlabel="Number of hits per cluster", pdf_name='clustersNHits.pdf', binning=list(range(0,50,1)), logY=True)

    def hist_2d(self, arr1, arr2, xlabel='blabla', ylabel='blabla', title='', pdf_name='blabla', logY=False, binning=None, annotate=None):
        fig, ax = plt.subplots()

        h = ax.hist2d(arr1, arr2, bins=binning)
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(self.save+pdf_name)


    def plot_hist1d(self, arr1, arr2, xlabel='blabla', pdf_name='blabla', logY=False, binning=None, annotate=None):

        plt.hist(arr1, histtype="step", bins=binning,
                color=self.c[0], linewidth=4, label=self.legend[0])
        plt.hist(arr2, histtype="step", bins=binning,
                color=self.c[1], linewidth=4, label=self.legend[1])  

        plt.grid(True)
        plt.xlabel(xlabel)
        hep.cms.text(text=configuration["cmsText"])
        plt.legend()
        if logY:
            plt.yscale('log')

        if annotate != None:
            plt.title(annotate)

        if (self.save != None):
            plt.savefig(self.save+pdf_name)

        plt.clf()

    def plot_hist1d_debug(self, arr1, xlabel='blabla', pdf_name='blabla', logY=False, binning=None, annotate=None):

        plt.hist(arr1, histtype="step", bins=binning,
                color=self.c[0], linewidth=4)

        plt.grid(True)
        plt.xlabel(xlabel)
        hep.cms.text(text=configuration["cmsText"])
        if logY:
            plt.yscale('log')

        if annotate != None:
            plt.title(annotate, loc='right', fontsize=12)
        if (self.save != None):
            plt.savefig(self.save+pdf_name)

        plt.clf()


    # FROM HERE: OLD CODE
    def nClustersPlot(self):
        nClustersPerEvent_lc = ak.count(self.lc_dict_arr['LayerClustersEnergy'], axis=1)
        nClustersPerEvent_lcpf = ak.count(self.lcpf_dict['LayerClustersEnergy'], axis=1)
        plt.hist(nClustersPerEvent_lc, histtype="step", range=(ak.min(nClustersPerEvent_lc), ak.max(nClustersPerEvent_lcpf)),
                color=self.c[0], linewidth=4, bins=20, label=self.legend[0])
        plt.hist(nClustersPerEvent_lcpf, histtype="step", range=(ak.min(nClustersPerEvent_lc), ak.max(nClustersPerEvent_lcpf)),
                color=self.c[1], linewidth=4, bins=20, label=self.legend[1])            
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
        plt.hist(ak.flatten(self.lc_dict_arr['LayerClustersEta']),
                color=self.c[0], histtype="step", linewidth=4, bins=20, label=self.legend[0])
        plt.hist(ak.flatten(self.lcpf_dict['LayerClustersEta']), color=self.c[1], histtype="step", linewidth=4, bins=20, label=self.legend[1])
        plt.grid(True)
        plt.xlabel(r"$\eta$")
        hep.cms.text(text=configuration["cmsText"])
        plt.legend()
        
        if (self.save != None):
            plt.savefig(self.save+"clustersEta.pdf")

        plt.clf()

    def clustersPhi(self):
        plt.hist(ak.flatten(self.lc_dict_arr['LayerClustersPhi']), color=self.c[0], histtype="step", linewidth=4, bins=20, label=self.legend[0])
        plt.hist(ak.flatten(self.lcpf_dict['LayerClustersPhi']), color=self.c[1], histtype="step", linewidth=4, bins=20, label=self.legend[1])
        plt.grid(True)
        plt.xlabel(r"$\phi$")
        hep.cms.text(text=configuration["cmsText"])
        plt.legend()
        
        if (self.save != None):
            plt.savefig(self.save+"clustersPhi.pdf")

        plt.clf()

    def clustersEnergy(self):
        plt.hist(ak.flatten(self.lc_dict_arr['LayerClustersEnergy']), color=self.c[0], histtype="step", linewidth=4, bins=20, label=self.legend[0])
        plt.hist(ak.flatten(self.lcpf_dict['LayerClustersEnergy']), color=self.c[1], histtype="step", linewidth=4, bins=20, label=self.legend[1])
        plt.grid(True)
        plt.xlabel("Energy [GeV]")
        plt.yscale('log')
        hep.cms.text(text=configuration["cmsText"])
        plt.legend()
        
        if (self.save != None):
            plt.savefig(self.save+"clustersEnergy.pdf")

        plt.clf()

    def clustersLayer(self):
        # TODO: check this
        plt.hist(ak.flatten(self.lc_dict_arr['LayerClustersLayer']), color=self.c[0], histtype="step", linewidth=4, bins=20, label=self.legend[0])
        plt.hist(ak.flatten(self.lcpf_dict['LayerClustersLayer']), color=self.c[1], histtype="step", linewidth=4, bins=20, label=self.legend[1])
        plt.grid(True)
        plt.xlabel("Layer")
        hep.cms.text(text=configuration["cmsText"])                                                                      
        plt.legend()
        
        if (self.save != None):
            plt.savefig(self.save+"clustersLayer.pdf")

        plt.clf()

    def nHitsPerCluster(self):

        plt.hist(ak.flatten(self.lc_dict_arr['LayerClustersNHits']), color=self.c[0], histtype="step", linewidth=4, bins=20, label=self.legend[0])
        plt.hist(ak.flatten(self.lcpf_dict['LayerClustersNHits']), color=self.c[1], histtype="step", linewidth=4, bins=20, label=self.legend[1])
        plt.grid(True)
        plt.xlabel("Number of hits per cluster")
        hep.cms.text(text=configuration["cmsText"])
        plt.legend()

        if (self.save != None):
            plt.savefig(self.save+"clustersNHits.pdf")
        
        plt.clf()
