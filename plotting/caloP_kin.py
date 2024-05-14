import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
from config.configuration import configuration
import awkward as ak
import numpy as np
import os

class CaloP_kin:
    def __init__(self, caloP, caloP_pf, lc_dict_arr=None, lcpf_dict=None, legend=None, c=["red", "black"], annotate=None,
                        save=None, recHits=None):
        self.caloP = caloP
        self.caloP_pf = caloP_pf
        self.legend = legend
        self.c = c
        self.annotate = annotate

        self.recHits = recHits
        self.lc_dict_arr = lc_dict_arr
        self.lcpf_dict_arr = lcpf_dict
        
        # iterate over all root files (which were saved in lc_dict_arr/lcpf_dict)
        for i in range(len(lc_dict_arr)):
            self.save = save + f'/file_{i}/CaloPPlots/'
            
            # Create the directory if doesn't exist
            if not os.path.exists(self.save):
                # Create the directory
                os.makedirs(self.save)            
                
            # plot results for current root file
            self.makePlot(self.lc_dict_arr[i], self.lcpf_dict_arr[i], self.caloP[i], self.caloP_pf[i], self.recHits[i])        
        
    def makePlot(self, lc_array, lcpf_array, caloP_array, caloP_pf_array, recHits_array):

        binsEta, _ = np.linspace(start=-1.5, stop=1.5, retstep=0.0175)
        binsEn = np.linspace(start=0, stop=50, num=11)
        binsPer = np.linspace(start=0, stop=1, num=11)
        binsPhi, _ = np.linspace(start=-np.pi, stop=np.pi, retstep=0.0175)

        # 2nd plot caloEta
        self.plot_hist1d(ak.flatten(caloP_array['caloParticleEta']),
                        ak.flatten(caloP_pf_array['caloParticleEta']),
                        xlabel=r"$\eta$", pdf_name='caloEta.png', binning=20, range=(-1.5, 1.5))

        # 3rd plot caloPhi
        self.plot_hist1d(ak.flatten(caloP_array['caloParticlePhi']),
                        ak.flatten(caloP_pf_array['caloParticlePhi']),
                        xlabel=r"$\phi$", pdf_name='caloPhi.png', binning=20, range=(-3.14, 3.14))

        # 4th plot caloEnergy
        self.plot_hist1d(ak.flatten(caloP_array['caloParticleEnergy']),
                        ak.flatten(caloP_pf_array['caloParticleEnergy']),
                        xlabel='Energy [GeV]', pdf_name='caloEnergy.png', logY=True, binning=10, range=(0,50))

        # 5th plot caloSharedEnergy
        self.plot_hist1d(ak.flatten(caloP_array['sharedEnergy'], axis=None),
                        ak.flatten(caloP_pf_array['sharedEnergy'], axis=None),
                        xlabel='Shared En [GeV]', pdf_name='caloSharedEn.png', binning=20, range=(0,60))

        # 6th plot caloSharedEnergy -> percentage
        self.plot_hist1d(ak.flatten(caloP_array['sharedEnergy']/caloP_array['caloParticleEnergy'], axis=None),
                        ak.flatten(caloP_pf_array['sharedEnergy']/caloP_pf_array['caloParticleEnergy'], axis=None),
                        xlabel='Shared En [%]', pdf_name='caloSharedEn_percentage.png', binning=20, range=(0,1))
        
        caloP_EnBroadcasted, y = ak.broadcast_arrays(caloP_array['caloParticleEnergy'], caloP_array['sharedEnergy'])
        
        self.hist_2d(ak.flatten(caloP_EnBroadcasted, axis=None).to_numpy(),
                    ak.flatten(caloP_array['sharedEnergy']/caloP_array['caloParticleEnergy'], axis=None).to_numpy(),
                    xlabel='caloP En [GeV]', ylabel='caloP shared En [%]', title=f'',
                    pdf_name='caloP_En_sharedEn_2D.png', logY=False, binning=(binsEn, binsPer))
        
        caloP_pf_EnBroadcasted, y = ak.broadcast_arrays(caloP_pf_array['caloParticleEnergy'], caloP_pf_array['sharedEnergy'])
        
        self.hist_2d(ak.flatten(caloP_pf_EnBroadcasted, axis=None).to_numpy(),
                    ak.flatten(caloP_pf_array['sharedEnergy']/caloP_pf_array['caloParticleEnergy'], axis=None).to_numpy(),
                    xlabel='caloP En [GeV]', ylabel='caloP shared En [%]', title=f'',
                    pdf_name='caloP_PF_En_sharedEn_2D.png', logY=False, binning=(binsEn, binsPer))
        
        caloP_sharedEn_sum = ak.sum(caloP_array['sharedEnergy'], axis=2)
        caloP_pf_sharedEn_sum = ak.sum(caloP_pf_array['sharedEnergy'], axis=2)
        
        self.hist_2d(ak.flatten(caloP_array['caloParticleEnergy'], axis=None).to_numpy(),
                    ak.flatten(caloP_sharedEn_sum/caloP_array['caloParticleEnergy'], axis=None).to_numpy(),
                    xlabel='caloP En [GeV]', ylabel='caloP shared En [%] (sum)', title=f'',
                    pdf_name='caloP_En_sharedEn_sum_2D.png', logY=False, binning=(binsEn, binsPer))
        
        self.hist_2d(ak.flatten(caloP_pf_array['caloParticleEnergy'], axis=None).to_numpy(),
                    ak.flatten(caloP_sharedEn_sum/caloP_pf_array['caloParticleEnergy'], axis=None).to_numpy(),
                    xlabel='caloP En [GeV]', ylabel='caloP shared En [%] (sum)', title=f'',
                    pdf_name='caloP_PF_En_sharedEn_sum_2D.png', logY=False, binning=(binsEn, binsPer))
        
        

        print(caloP_array['sharedEnergy'][:5]/caloP_array['caloParticleEnergy'][:5])
        print(caloP_array['simToRecoAssociation'][:5])
        print(caloP_pf_array['sharedEnergy'][:5]/caloP_pf_array['caloParticleEnergy'][:5])
        print(caloP_pf_array['simToRecoAssociation'][:5])
        print()
        

        
        if False:
            lcpf_array['recoToSimAssociation'] = ak.pad_none(lcpf_array['recoToSimAssociation'], 1, axis=2)
            # TO DISCUSS: THERE ARE SOME LC with reco2Sim Null (it doesn't exist)
            lcpf_array['recoToSimAssociation'] = ak.fill_none(lcpf_array['recoToSimAssociation'], 2)
            lcpf_array['recoToSimAssociation'] = ak.flatten(lcpf_array['recoToSimAssociation'], axis=2)

            phi_diff = lcpf_array['layerClusterPhi'] - caloP_array['caloParticlePhi'][:, np.newaxis]
            phi_diff_correct = np.pi - abs(abs(phi_diff) - np.pi)

            deltaR_caloP_PF_all = (lcpf_array['layerClusterEta'] - caloP_array['caloParticleEta'][:, np.newaxis])**2 + \
                                phi_diff_correct**2

            print(ak.flatten(deltaR_caloP_PF_all, axis=None).to_numpy().shape)
            print(ak.flatten(lcpf_array['recoToSimAssociation'], axis=None).to_numpy().shape)

            self.hist_2d(np.sqrt(ak.flatten(deltaR_caloP_PF_all, axis=None).to_numpy()),
                        ak.flatten(lcpf_array['recoToSimAssociation'], axis=None).to_numpy(),
                        xlabel='delta R (PF - CaloP)', ylabel='reco2Sim score', title=f'no cut',
                        pdf_name='reco2Sim_deltaR_2D.png', logY=False, binning=200)

            # reco-to-sim high
            self.plot_hist1d(ak.flatten(lc_array['recoToSimAssociation'], axis=None).to_numpy(),
                            ak.flatten(lcpf_array['recoToSimAssociation'], axis=None).to_numpy(),
                            xlabel="reco-to-sim score", pdf_name='reco2SimScore.png', binning=30, logY=True, range=(0,3))

            min_reco_to_sim = 1.0
            mask_reco_to_sim = lcpf_array['recoToSimAssociation'] > min_reco_to_sim
            mask_reco_to_sim_perEvent = ak.any(mask_reco_to_sim, axis=1) # check if there is any true per event

            self.plot_hist1d_debug(ak.flatten(lcpf_array['recoToSimAssociation'][mask_reco_to_sim]),
                                    xlabel='reco2Sim score', pdf_name='reco2Sim_cut.png', logY=True, binning=None,
                                    annotate=f'PF clusters where reco2Sim > {min_reco_to_sim}')

            self.plot_hist1d_debug(ak.flatten(lcpf_array['layerClusterEnergy'][mask_reco_to_sim]),
                                    xlabel='Energy [GeV]', pdf_name='reco2Sim_PFEnergy.png', logY=True, binning=None,
                                    annotate=f'PF clusters where reco2Sim > {min_reco_to_sim}')

            self.plot_hist1d_debug(ak.flatten(lcpf_array['layerClusterNumberOfHits'][mask_reco_to_sim]),
                                    xlabel='Number of hits per cluster', pdf_name='reco2Sim_PFNHits.png', logY=True, binning=None,
                                    annotate=f'PF clusters where reco2Sim > {min_reco_to_sim}')

            self.hist_2d(ak.flatten(lcpf_array['layerClusterEta'][mask_reco_to_sim]).to_numpy(),
                        ak.flatten(lcpf_array['layerClusterPhi'][mask_reco_to_sim]).to_numpy(),
                        xlabel='PF ETA', ylabel='PF PHI', title=f'PF clusters where reco2Sim > {min_reco_to_sim}',
                        pdf_name='reco2Sim_PFEtaPhi.png', logY=False, binning=(binsEta, binsPhi))

            self.hist_2d(ak.flatten(lcpf_array['layerClusterEta'][mask_reco_to_sim]).to_numpy(),
                        ak.flatten(lcpf_array['layerClusterEnergy'][mask_reco_to_sim]).to_numpy(),
                        xlabel='PF ETA', ylabel='PF Energy [GeV]', title=f'PF where reco2Sim > {min_reco_to_sim}',
                        pdf_name='reco2Sim_PFEtaEn.png', logY=False, binning=(binsEta, [0,2,4,6,8,10]))

            phi_diff_maskedReco2Sim = lcpf_array['layerClusterPhi'][mask_reco_to_sim] - caloP_array['caloParticlePhi'][mask_reco_to_sim_perEvent][:, np.newaxis]
            phi_diff_maskedReco2Sim_correct = np.pi - abs(abs(phi_diff_maskedReco2Sim) - np.pi)

            deltaR_caloP_PF = (lcpf_array['layerClusterEta'][mask_reco_to_sim] - caloP_array['caloParticleEta'][mask_reco_to_sim_perEvent][:, np.newaxis])**2 + \
                                phi_diff_maskedReco2Sim_correct**2

            deltaR_caloP_PF = ak.flatten(deltaR_caloP_PF, axis=1)

            self.plot_hist1d_debug(np.sqrt(ak.flatten(deltaR_caloP_PF).to_numpy()),
                                    xlabel='delta R (PF - CaloP)', pdf_name='reco2Sim_PFDeltaR.png', logY=True, binning=None,
                                    annotate=f'PF where reco2Sim > {min_reco_to_sim}')

            self.hist_2d(ak.flatten(lcpf_array['layerClusterEta'][mask_reco_to_sim]).to_numpy(),
                        np.sqrt(ak.flatten(deltaR_caloP_PF).to_numpy()),
                        xlabel='PF ETA', ylabel='delta R (PF - CaloP)', title=f'PF where reco2Sim > {min_reco_to_sim}',
                        pdf_name='reco2Sim_PFEtaDeltaR.png', logY=False, binning=(binsEta, [0,1,2,3,4,5]))

            self.hist_2d(ak.flatten(lcpf_array['layerClusterEnergy'][mask_reco_to_sim]).to_numpy(),
                        np.sqrt(ak.flatten(deltaR_caloP_PF).to_numpy()),
                        xlabel='PF En [GeV]', ylabel='delta R (PF - CaloP)', title=f'PF where reco2Sim > {min_reco_to_sim}',
                        pdf_name='reco2Sim_PFEnDeltaR.png', logY=False, binning=40)

            # check reco to sim and distance for PF clusters > 4 GeV
            min_En = 4
            mask_PFCluster_En = lcpf_array['layerClusterEnergy'] > min_En
            mask_PFCluster_En_perEvent = ak.any(mask_PFCluster_En, axis=1) # check if there is any true per event

            # self.lcpf_dict_arr['layerClusterPhi'][mask_PFCluster_En][mask_PFCluster_En_perEvent] -> because
            # without the last mask (perEvent) -> there are still left some events with 0 PF clusters: []
            # to remove these empty lists -> attach last mask
            phi_diff_EnCut = lcpf_array['layerClusterPhi'][mask_PFCluster_En][mask_PFCluster_En_perEvent] - caloP_array['caloParticlePhi'][mask_PFCluster_En_perEvent][:, np.newaxis]
            phi_diff_EnCut_correct = np.pi - abs(abs(phi_diff_EnCut) - np.pi)

            deltaR_caloP_PF_EnCut = (lcpf_array['layerClusterEta'][mask_PFCluster_En][mask_PFCluster_En_perEvent] - caloP_array['caloParticleEta'][mask_PFCluster_En_perEvent][:, np.newaxis])**2 + \
                                phi_diff_EnCut_correct**2

            self.hist_2d(np.sqrt(ak.flatten(deltaR_caloP_PF_EnCut, axis=None).to_numpy()),
                        ak.flatten(lcpf_array['recoToSimAssociation'][mask_PFCluster_En][mask_PFCluster_En_perEvent]).to_numpy(),
                        xlabel='delta R (PF - CaloP)', ylabel='reco2Sim score', title=f'PF where En > {min_En} GeV',
                        pdf_name='reco2Sim_deltaR_2D_EnCut.png', logY=False, binning=200)

            self.plot_hist1d_debug(np.sqrt(ak.flatten(deltaR_caloP_PF_EnCut).to_numpy()),
                                    xlabel='delta R (PF - CaloP)', pdf_name='reco2Sim_PFDeltaR_EnCut.png', logY=True, binning=None,
                                    annotate=f'PF where En > {min_En} GeV')



    def hist_2d(self, arr1, arr2, xlabel='blabla', ylabel='blabla', title='', pdf_name='blabla', logY=False, binning=None):
        fig, ax = plt.subplots()

        h = ax.hist2d(arr1, arr2, bins=binning, cmin=1, norm=mpl.colors.LogNorm())
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        hep.cms.text(text=configuration["cmsText"])
        
        ax.set_title(title, loc='right', fontsize=12)
        
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
            plt.title(annotate, loc='right', fontsize=12)

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

