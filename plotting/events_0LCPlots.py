import matplotlib.pyplot as plt
import mplhep as hep
from config.configuration import configuration
import awkward as ak
import numpy as np
from pathlib import Path
import os

class Events_0LCPlots:
    def __init__(self, lc_dict_arr, lcpf_dict, legend=None, c=["red", "black"], annotate=None,
                        save=None, caloP=None, recHits=None):
        self.lc_dict_arr = lc_dict_arr
        self.lcpf_dict_arr = lcpf_dict
        self.legend = legend
        self.c = c
        self.annotate = annotate
        self.save = save + '/events_0LC/'
        self.caloP = caloP
        self.recHits = recHits

        # Create the directory
        if not os.path.exists(self.save):
                # Create the directory
                os.makedirs(self.save)

        self.makePlot()
        
    def makePlot(self):
        # 1st plot nClustersPerEvent
        nClustersPerEvent_lc = ak.count(self.lc_dict_arr['layerClusterEnergy'], axis=1)
        nClustersPerEvent_lcpf = ak.count(self.lcpf_dict_arr['layerClusterEnergy'], axis=1)
        

        binsEta, _ = np.linspace(start=-1.5, stop=1.5, retstep=0.0175)
        binsPhi, _ = np.linspace(start=-np.pi, stop=np.pi, retstep=0.0175)

        clustersEta = ak.flatten(self.lc_dict_arr['layerClusterEta']).to_numpy()
        clustersPhi = ak.flatten(self.lc_dict_arr['layerClusterPhi']).to_numpy()
        self.hist_2d(clustersEta, clustersPhi, xlabel='LC ETA', ylabel='LC PHI',
                title='',
                pdf_name='LC_eta_phi.png', logY=False, binning=(binsEta, binsPhi))

        mask_noLc_0 = nClustersPerEvent_lc == 0
        caloEta_noLc_0 = ak.flatten(self.caloP['caloParticleEta'][mask_noLc_0]).to_numpy()
        caloPhi_noLc_0 = ak.flatten(self.caloP['caloParticlePhi'][mask_noLc_0]).to_numpy()
        self.hist_2d(caloEta_noLc_0, caloPhi_noLc_0, xlabel='CaloP ETA',
                ylabel='CaloP PHI', title='CaloP when #LC = 0',
                pdf_name='eta_phi_noLC_0.png', logY=False, binning=(binsEta, binsPhi))

        self.hist_2d(caloEta_noLc_0, ak.flatten(self.caloP['caloParticleEnergy'][mask_noLc_0]).to_numpy(), xlabel='CaloP ETA',
                ylabel='CaloP Energy', title='CaloP when #LC = 0',
                pdf_name='eta_En_noLC_0.png', logY=False,
                binning=(binsEta, np.array([50,75,100,150,200,250,350,500])))


        recHitsEta = ak.flatten(self.recHits['pfrechitEta']).to_numpy()
        recHitsPhi = ak.flatten(self.recHits['pfrechitPhi']).to_numpy()

        self.hist_2d(recHitsEta, recHitsPhi, xlabel='RecHits ETA',
                ylabel='RecHits PHI', title='',
                pdf_name='eta_phi_recHits.png', logY=False, binning=(binsEta, binsPhi))

        recHitsEta_noLc_0 = ak.flatten(self.recHits['pfrechitEta'][mask_noLc_0]).to_numpy()
        recHitsPhi_noLc_0 = ak.flatten(self.recHits['pfrechitPhi'][mask_noLc_0]).to_numpy()
        self.hist_2d(recHitsEta_noLc_0, recHitsPhi_noLc_0,
                xlabel='RecHits ETA', ylabel='RecHits PHI', title='RecHits when #LC = 0',
                pdf_name='eta_phi_recHits_noLc_0.png', logY=False, binning=(binsEta, binsPhi))
        
        self.plot_hist1d_debug(ak.flatten(self.caloP['caloParticleEnergy'][mask_noLc_0]),
                xlabel='Energy', pdf_name='caloParticlesEnergy_noLc_0.png')
        self.plot_hist1d_debug(ak.flatten(self.caloP['caloParticleEta'][mask_noLc_0]),
                xlabel=r'\eta', pdf_name='caloParticlesEta_noLc_0.png')
        self.plot_hist1d_debug(ak.flatten(self.caloP['caloParticlePhi'][mask_noLc_0]),
                xlabel=r'\phi', pdf_name='caloParticlesPhi_noLc_0.png')

        #x = self.caloP['caloParticleEnergy'][mask_noLc_0]
        #y = self.caloP['caloParticleEta'][mask_noLc_0]
        CaloPhi_masked = self.caloP['caloParticlePhi'][mask_noLc_0]

        No_total_ev = len(self.lc_dict_arr['layerClusterNumberOfHits'])

        En_cut = 50
        mask_noLC_0_EnCut = self.caloP['caloParticleEnergy'][mask_noLc_0] > En_cut
        no_events_noLc_0_EnCut = ak.count_nonzero(mask_noLC_0_EnCut)

        string0 = f'1st cut: # LC/event = 0 \n 2nd cut: En of CaloP > {En_cut} \n {no_events_noLc_0_EnCut}/{No_total_ev} events'
        self.plot_hist1d_debug(ak.flatten(self.caloP['caloParticleEnergy'][mask_noLc_0][mask_noLC_0_EnCut]), xlabel='CaloParticle Energy',
                            pdf_name='caloParticlesEnergy_NoLC_0_EnCut.png', annotate=string0)
        self.plot_hist1d_debug(ak.flatten(self.caloP['caloParticleEta'][mask_noLc_0][mask_noLC_0_EnCut]), xlabel=r'CaloParticle \eta',
                            pdf_name='caloParticlesEta_NoLC_0_EnCut.png', annotate=string0)

        eta_cut = 0.2
        mask_noLC_0_EtaCut = abs(self.caloP['caloParticleEta'][mask_noLc_0]) > eta_cut
        no_events_noLC_0_EtaCut = np.count_nonzero(mask_noLC_0_EtaCut)
        string1 = f'1st cut: # LC/event = 0 \n 2nd cut: abs(eta) of CaloP > {eta_cut} \n {no_events_noLC_0_EtaCut}/{No_total_ev} events'
        self.plot_hist1d_debug(ak.flatten(self.caloP['caloParticleEnergy'][mask_noLc_0][mask_noLC_0_EtaCut]), xlabel='CaloParticle Energy',
                            pdf_name='caloParticlesEnergy_NoLC_0_EtaCut.png', annotate=string1)
        self.plot_hist1d_debug(ak.flatten(self.caloP['caloParticleEta'][mask_noLc_0][mask_noLC_0_EtaCut]), xlabel=r'CaloParticle \eta',
                            pdf_name='caloParticlesEta_NoLC_0_EtaCut.png', annotate=string1)


    def hist_2d(self, arr1, arr2, xlabel='blabla', ylabel='blabla', title='', pdf_name='blabla', logY=False, binning=None, annotate=None):
        fig, ax = plt.subplots()

        h = ax.hist2d(arr1, arr2, bins=binning, cmin=1)
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(self.save+pdf_name)
        plt.close(fig)


    def plot_hist1d(self, arr1, arr2, xlabel='blabla', pdf_name='blabla', range=None, logY=False, binning=None, annotate=None):

        plt.hist(arr1, histtype="step", bins=binning,
                color=self.c[0], linewidth=4, label=self.legend[0], range=range)
        plt.hist(arr2, histtype="step", bins=binning,
                color=self.c[1], linewidth=4, label=self.legend[1], range=range)  

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