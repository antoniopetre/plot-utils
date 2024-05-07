import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from utils.makeRatio import makeRatio
from utils.handleUncertainties import handleUncertainties
from config.configuration import configuration
import awkward as ak
import numpy as np
import os
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
from math import sqrt

class EfficiencyPlots:
    def __init__ (self, 
                  lc_dict_arr, cp_dict_arr,
                  cut_sim2Reco=[1.0],
                  annotate=None,
                  legend=None,
                  c=['red', 'black'],
                  save=None):

        self.annotate=annotate
        self.legend=legend
        self.c = c
        
        # case when we don't compare with pf efficiency
        if len(cp_dict_arr) == 1:
            
            self.lc_dict_arr = lc_dict_arr[0]
            self.cp_dict_arr = cp_dict_arr[0]

            # iterate over all thresholds
            for cut_sim2Reco_elem in cut_sim2Reco:

                self.sim2Reco_cut = cut_sim2Reco_elem

                # iterate over all root files (which were saved in lc_dict_arr/lcpf_dict)
                for i in range(len(self.lc_dict_arr)):
                    self.save = save + f'/file_{i}/efficiency_plots_cutSim2Reco:{self.sim2Reco_cut}/'

                    # Create the directory if doesn't exist
                    if not os.path.exists(self.save):
                        # Create the directory
                        os.makedirs(self.save)            

                    # plot results for current root file
                    self.makePlot(self.cp_dict_arr[i]) 
        
        # case when we compare with pf efficiency
        elif len(cp_dict_arr) == 2:
            
            self.lc_dict_arr = lc_dict_arr[0]
            self.cp_dict_arr = cp_dict_arr[0]
            self.lcpf_dict_arr = lc_dict_arr[1]
            self.cppf_dict_arr = cp_dict_arr[1]
            
            # iterate over all thresholds
            for cut_sim2Reco_elem in cut_sim2Reco:

                self.sim2Reco_cut = cut_sim2Reco_elem

                # iterate over all root files (which were saved in lc_dict_arr/lcpf_dict)
                for i in range(len(self.lc_dict_arr)):
                    self.save = save + f'/file_{i}/efficiency_plots_comparisonPF_cutSim2Reco:{self.sim2Reco_cut}/'

                    # Create the directory if doesn't exist
                    if not os.path.exists(self.save):
                        # Create the directory
                        os.makedirs(self.save)            

                    # plot results for current root file
                    self.makePlot_comparison(self.cp_dict_arr[i], self.cppf_dict_arr[i]) 
            
        
    def compute_eff_perBin(self, caloP_kinematics, noAssociatedLC_perEv, bins):
        
        efficiencyPerBin = [-1] * (len(bins) - 1)
        errPerBin = [-1] * (len(bins) - 1)
        
        for i in range(len(bins) - 1):
            mask_lower = caloP_kinematics < bins[i+1]
            mask_higher = caloP_kinematics > bins[i]
            
            mask_Events = np.logical_and(mask_lower.to_numpy(), mask_higher.to_numpy())
                        
            mask_AssociatedLC = noAssociatedLC_perEv[mask_Events] > 0
            efficiencyPerBin[i] = ak.count_nonzero(mask_AssociatedLC, axis=0)/ak.num(mask_AssociatedLC, axis=0)
            # error = sqrt(N_good)/N_total
            errPerBin[i] = sqrt(ak.count_nonzero(mask_AssociatedLC, axis=0))/ak.num(mask_AssociatedLC, axis=0)
            
        return efficiencyPerBin, errPerBin
    
    def compute_eff_perBin_2d(self, caloP_kinematics_1, caloP_kinematics_2, noAssociatedLC_perEv, bins_1, bins_2):
        
        efficiencyPerBin = np.array([[-1] * (len(bins_2) - 1)] * (len(bins_1) - 1), dtype=np.float64)
                
        for i in range(len(bins_1) - 1):
            mask_lower_1 = caloP_kinematics_1 < bins_1[i+1]
            mask_higher_1 = caloP_kinematics_1 > bins_1[i]
            
            mask_Events_1 = np.logical_and(mask_lower_1.to_numpy(), mask_higher_1.to_numpy())
            
            
            for j in range(len(bins_2) - 1):
                mask_lower_2 = caloP_kinematics_2 < bins_2[j+1]
                mask_higher_2 = caloP_kinematics_2 > bins_2[j]

                mask_Events_2 = np.logical_and(mask_lower_2.to_numpy(), mask_higher_2.to_numpy())
                                
                mask_Events_final = np.logical_and(mask_Events_1, mask_Events_2)
                                
                mask_AssociatedLC = noAssociatedLC_perEv[mask_Events_final] > 0
                
                efficiencyPerBin[i,j] = ak.count_nonzero(mask_AssociatedLC, axis=0)/ak.num(mask_AssociatedLC, axis=0)
            
        efficiencyPerBin = np.transpose(efficiencyPerBin, (1,0)) # do transpose because heatmap need the data passed row by row
        
        return efficiencyPerBin
            
            

    def makePlot(self, caloP_array):
        
        
        mask = caloP_array['simToRecoAssociation'] == 1
        mask_perEvent = ak.any(ak.flatten(mask, axis=1), axis=1)
        
        self.plot_hist1d_debug(ak.flatten(caloP_array['simToRecoAssociation'], axis=None),
                               xlabel='sim2Reco', pdf_name='caloP_sim2reco.png', logY=True, binning=100, annotate=None)
        
        self.plot_hist1d_debug(ak.flatten(caloP_array['simToRecoAssociation'][~mask_perEvent], axis=None),
                               xlabel='sim2Reco', pdf_name=f'caloP_sim2reco_CutAt_{self.sim2Reco_cut}.png', logY=True, binning=100, annotate=None)
                
        mask_sim2Reco_cut = caloP_array['simToRecoAssociation'] <= self.sim2Reco_cut
                    
        noAssociatedLC_perEv = ak.num(caloP_array['AssociatedLC'][mask_sim2Reco_cut], axis=2)
                
        En_bins = np.linspace(0, 50, 11)
        Eta_bins = np.linspace(-1.5, 1.5, 11)
        Phi_bins = np.linspace(-np.pi, np.pi, 11)
        PT_bins = np.linspace(0, 50, 11)
        
        caloP_pt = caloP_array['caloParticleEnergy'].to_numpy()/np.cosh(caloP_array['caloParticleEta'].to_numpy())
        
        efficiency_En, errPerEn = self.compute_eff_perBin(caloP_array['caloParticleEnergy'], noAssociatedLC_perEv, En_bins)
        efficiency_Eta, errPerEta = self.compute_eff_perBin(caloP_array['caloParticleEta'], noAssociatedLC_perEv, Eta_bins)
        efficiency_Phi, errPerPhi = self.compute_eff_perBin(caloP_array['caloParticlePhi'], noAssociatedLC_perEv, Phi_bins)
        efficiency_PT, errPerPT = self.compute_eff_perBin(ak.Array(caloP_pt), noAssociatedLC_perEv, PT_bins)
        
        self.plot_bar_efficiency(efficiency_En, errPerEn, En_bins, xlabel='Energy [GeV]', title='', ylim=(0,1.05), pdf_name='efficiency_energy.png', logY=False)
        self.plot_bar_efficiency(efficiency_Eta, errPerEta, Eta_bins, xlabel='eta', title='', ylim=(0,1.05), pdf_name='efficiency_eta.png', logY=False)
        self.plot_bar_efficiency(efficiency_Phi, errPerPhi, Phi_bins, xlabel='phi', title='', ylim=(0,1.05), pdf_name='efficiency_phi.png', logY=False)
        self.plot_bar_efficiency(efficiency_PT, errPerPT, PT_bins, xlabel='pt [GeV]', title='', ylim=(0,1.05), pdf_name='efficiency_pt.png', logY=False)
        
        En_bins = np.linspace(0, 50, 6)
        Eta_bins = np.linspace(-1.5, 1.5, 6)
        Phi_bins = np.linspace(-np.pi, np.pi, 6)
        PT_bins = np.linspace(0, 50, 6)
        
        efficiency_En_eta = self.compute_eff_perBin_2d(caloP_array['caloParticleEnergy'], caloP_array['caloParticleEta'], noAssociatedLC_perEv, En_bins, Eta_bins)
        efficiency_En_phi = self.compute_eff_perBin_2d(caloP_array['caloParticleEnergy'], caloP_array['caloParticlePhi'], noAssociatedLC_perEv, En_bins, Phi_bins)
        efficiency_En_pt = self.compute_eff_perBin_2d(caloP_array['caloParticleEnergy'], ak.Array(caloP_pt), noAssociatedLC_perEv, En_bins, PT_bins)
        efficiency_Phi_eta = self.compute_eff_perBin_2d(caloP_array['caloParticlePhi'], caloP_array['caloParticleEta'], noAssociatedLC_perEv, Phi_bins, Eta_bins)
        
        self.colormap(efficiency_En_eta, pdf_name='efficiency_EnEta2D.png', xlabel='En', ylabel='eta', xticks=En_bins, yticks=Eta_bins)
        self.colormap(efficiency_En_phi, pdf_name='efficiency_EnPhi2D.png', xlabel='En', ylabel='phi', xticks=En_bins, yticks=Phi_bins)
        self.colormap(efficiency_En_pt, pdf_name='efficiency_EnPt2D.png', xlabel='En', ylabel='pt [GeV]', xticks=En_bins, yticks=PT_bins)
        self.colormap(efficiency_Phi_eta, pdf_name='efficiency_PhiEta2D.png', xlabel='phi', ylabel='eta', xticks=Phi_bins, yticks=Eta_bins)
        
        
        if False:
            fig, axs = plt.subplots(1, 3, figsize=(30, 15))

            for cp_dict, k in zip(self.cp_dict_arr, range(0, len(self.cp_dict_arr))):
                if (cumulative):
                    eta_eff, phi_eff, energy_eff = countCumulative(cp_dict)
                else:
                    eta_eff, phi_eff, energy_eff = count(cp_dict)

                num_eta_counts, _ = np.histogram(ak.flatten(eta_eff[0]), bins=self.eta_bins)            
                denom_eta_counts, _ = np.histogram(ak.flatten(eta_eff[1]), bins=self.eta_bins)            

                eta_ratio, eta_ratio_error = makeRatio(num_eta_counts, denom_eta_counts)
                eta_ratio_error = handleUncertainties(eta_ratio, eta_ratio_error)
                axs[0].errorbar(self.eta_bins[:-1], \
                                eta_ratio, yerr=eta_ratio_error, \
                                fmt=".", markersize=20, color=self.c[k], \
                                label=self.legend[k])
                axs[0].set_xlabel(r"$\eta$")
                axs[0].set_xlim(-1.2, 1.2) 

                num_phi_counts, _ = np.histogram(ak.flatten(phi_eff[0]), bins=self.phi_bins)            
                denom_phi_counts, _ = np.histogram(ak.flatten(phi_eff[1]), bins=self.phi_bins)            

                phi_ratio, phi_ratio_error = makeRatio(num_phi_counts, denom_phi_counts)
                phi_ratio_error = handleUncertainties(phi_ratio, phi_ratio_error)
                axs[1].errorbar(self.phi_bins[:-1], \
                                phi_ratio, yerr=phi_ratio_error, \
                                fmt=".", markersize=20, color=self.c[k], \
                                label=self.legend[k])
                axs[1].set_xlabel(r"$\phi$")
                axs[1].set_xlim(-np.pi, np.pi) 

                num_energy_counts, _ = np.histogram(ak.flatten(energy_eff[0]), bins=self.energy_bins)            
                denom_energy_counts, _ = np.histogram(ak.flatten(energy_eff[1]), bins=self.energy_bins)            

                energy_ratio, energy_ratio_error = makeRatio(num_energy_counts, denom_energy_counts)
                energy_ratio_error = handleUncertainties(energy_ratio, energy_ratio_error)
                axs[2].errorbar(self.energy_bins[:-1], \
                                energy_ratio, yerr=energy_ratio_error, \
                                fmt=".", markersize=20, color=self.c[k], \
                                label=self.legend[k])
                axs[2].set_xlabel(r"Energy (sim) [GeV]")

            for ax in axs:
                hep.cms.text(text=configuration["cmsText"], ax=ax)
                ax.set_ylim(-1e-2, 1.1)
                ax.set_ylabel('Efficiency')
                ax.grid(True)
                ax.legend()
                if self.annotate != None:
                    if "Energy (sim) [GeV]" not in ax.get_xlabel():
                        ax.annotate(text=self.annotate, xy=(ax.get_xlim()[0]*0.9, 0.4), fontsize=20)
                    else:
                        ax.annotate(text=self.annotate, xy=(ax.get_xlim()[1]*0.2, 0.4), fontsize=20)

            if (cumulative) :
                if (self.save != None and len(self.c) == 2):
                    fig.savefig(self.save+"efficiency_comparison_cumulative.png")
                elif (self.save != None):
                    fig.savefig(self.save+"efficiency_cumulative.png")
            else:
                if (self.save != None and len(self.c) == 2):
                    fig.savefig(self.save+"efficiency_comparison.png")
                elif (self.save != None):
                    fig.savefig(self.save+"efficiency.png")


            fig.clf()
            
    def makePlot_comparison(self, caloP_array, caloP_pf_array):
        
        
        mask = caloP_array['simToRecoAssociation'] <= self.sim2Reco_cut
        mask_perEvent = ak.any(ak.flatten(mask, axis=1), axis=1)
        
        mask_pf = caloP_pf_array['simToRecoAssociation'] <= self.sim2Reco_cut
        mask_pf_perEvent = ak.any(ak.flatten(mask_pf, axis=1), axis=1)
        
        self.plot_hist1d_debug(ak.flatten(caloP_array['simToRecoAssociation'], axis=None),
                               xlabel='sim2Reco', pdf_name='caloP_sim2reco.png', logY=True, binning=100, annotate=None)
        
        self.plot_hist1d_debug(ak.flatten(caloP_array['simToRecoAssociation'][~mask_perEvent], axis=None),
                               xlabel='sim2Reco', pdf_name=f'caloP_sim2reco_CutAt_{self.sim2Reco_cut}.png', logY=True, binning=100, annotate=None)
        
        self.plot_hist1d_debug(ak.flatten(caloP_pf_array['simToRecoAssociation'], axis=None),
                               xlabel='sim2Reco PF', pdf_name='caloP_sim2reco_PF.png', logY=True, binning=100, annotate=None)
        
        self.plot_hist1d_debug(ak.flatten(caloP_pf_array['simToRecoAssociation'][~mask_pf_perEvent], axis=None),
                               xlabel='sim2Reco PF', pdf_name=f'caloP_sim2reco_PF_CutAt_{self.sim2Reco_cut}.png', logY=True, binning=100, annotate=None)
                
        mask_sim2Reco_cut = caloP_array['simToRecoAssociation'] <= self.sim2Reco_cut
        mask_sim2Reco_PF_cut = caloP_pf_array['simToRecoAssociation'] <= self.sim2Reco_cut

        print(caloP_array['sharedEnergy'][mask_sim2Reco_cut])
        print(caloP_array['caloParticleEnergy'])
        print(caloP_array['simToRecoAssociation'][mask_sim2Reco_cut])

        exit(0)
                    
        noAssociatedLC_perEv = ak.num(caloP_array['AssociatedLC'][mask_sim2Reco_cut], axis=2)
        noAssociatedLC_perEv_PF = ak.num(caloP_pf_array['AssociatedLC'][mask_sim2Reco_PF_cut], axis=2)
                
        En_bins = np.linspace(0, 50, 11)
        Eta_bins = np.linspace(-1.5, 1.5, 11)
        Phi_bins = np.linspace(-np.pi, np.pi, 11)
        PT_bins = np.linspace(0, 50, 11)
        
        caloP_pt = caloP_array['caloParticleEnergy'].to_numpy()/np.cosh(caloP_array['caloParticleEta'].to_numpy())
        caloP_pt_PF = caloP_pf_array['caloParticleEnergy'].to_numpy()/np.cosh(caloP_pf_array['caloParticleEta'].to_numpy())
        
        efficiency_En, errPerEn = self.compute_eff_perBin(caloP_array['caloParticleEnergy'], noAssociatedLC_perEv, En_bins)
        efficiency_Eta, errPerEta = self.compute_eff_perBin(caloP_array['caloParticleEta'], noAssociatedLC_perEv, Eta_bins)
        efficiency_Phi, errPerPhi = self.compute_eff_perBin(caloP_array['caloParticlePhi'], noAssociatedLC_perEv, Phi_bins)
        efficiency_PT, errPerPT = self.compute_eff_perBin(ak.Array(caloP_pt), noAssociatedLC_perEv, PT_bins)
        
        efficiency_En_PF, errPerEn_PF = self.compute_eff_perBin(caloP_pf_array['caloParticleEnergy'], noAssociatedLC_perEv_PF, En_bins)
        efficiency_Eta_PF, errPerEta_PF = self.compute_eff_perBin(caloP_pf_array['caloParticleEta'], noAssociatedLC_perEv_PF, Eta_bins)
        efficiency_Phi_PF, errPerPhi_PF = self.compute_eff_perBin(caloP_pf_array['caloParticlePhi'], noAssociatedLC_perEv_PF, Phi_bins)
        efficiency_PT_PF, errPerPT_PF = self.compute_eff_perBin(ak.Array(caloP_pt_PF), noAssociatedLC_perEv_PF, PT_bins)
        
        self.plot_bar_efficiency(efficiency_En, errPerEn, En_bins, xlabel='Energy [GeV]', title='', ylim=(0,1.05), pdf_name='efficiency_energy.png', logY=False, efficiency_pf=efficiency_En_PF, efficiency_err_pf=errPerEn_PF)
        self.plot_bar_efficiency(efficiency_Eta, errPerEta, Eta_bins, xlabel='eta', title='', ylim=(0,1.05), pdf_name='efficiency_eta.png', logY=False, efficiency_pf=efficiency_Eta_PF, efficiency_err_pf=errPerEta_PF)
        self.plot_bar_efficiency(efficiency_Phi, errPerPhi, Phi_bins, xlabel='phi', title='', ylim=(0,1.05), pdf_name='efficiency_phi.png', logY=False, efficiency_pf=efficiency_Phi_PF, efficiency_err_pf=errPerPhi_PF)
        self.plot_bar_efficiency(efficiency_PT, errPerPT, PT_bins, xlabel='pt [GeV]', title='', ylim=(0,1.05), pdf_name='efficiency_pt.png', logY=False, efficiency_pf=efficiency_PT_PF, efficiency_err_pf=errPerPT_PF)
        
        En_bins = np.linspace(0, 50, 6)
        Eta_bins = np.linspace(-1.5, 1.5, 6)
        Phi_bins = np.linspace(-np.pi, np.pi, 6)
        PT_bins = np.linspace(0, 50, 6)
        
        efficiency_En_eta = self.compute_eff_perBin_2d(caloP_array['caloParticleEnergy'], caloP_array['caloParticleEta'], noAssociatedLC_perEv, En_bins, Eta_bins)
        efficiency_En_phi = self.compute_eff_perBin_2d(caloP_array['caloParticleEnergy'], caloP_array['caloParticlePhi'], noAssociatedLC_perEv, En_bins, Phi_bins)
        efficiency_En_pt = self.compute_eff_perBin_2d(caloP_array['caloParticleEnergy'], ak.Array(caloP_pt), noAssociatedLC_perEv, En_bins, PT_bins)
        efficiency_Phi_eta = self.compute_eff_perBin_2d(caloP_array['caloParticlePhi'], caloP_array['caloParticleEta'], noAssociatedLC_perEv, Phi_bins, Eta_bins)
        
        efficiency_En_eta_PF = self.compute_eff_perBin_2d(caloP_pf_array['caloParticleEnergy'], caloP_pf_array['caloParticleEta'], noAssociatedLC_perEv_PF, En_bins, Eta_bins)
        efficiency_En_phi_PF = self.compute_eff_perBin_2d(caloP_pf_array['caloParticleEnergy'], caloP_pf_array['caloParticlePhi'], noAssociatedLC_perEv_PF, En_bins, Phi_bins)
        efficiency_En_pt_PF = self.compute_eff_perBin_2d(caloP_pf_array['caloParticleEnergy'], ak.Array(caloP_pt_PF), noAssociatedLC_perEv_PF, En_bins, PT_bins)
        efficiency_Phi_eta_PF = self.compute_eff_perBin_2d(caloP_pf_array['caloParticlePhi'], caloP_pf_array['caloParticleEta'], noAssociatedLC_perEv_PF, Phi_bins, Eta_bins)
        
        self.colormap(efficiency_En_eta, pdf_name='efficiency_EnEta2D.png', xlabel='En', ylabel='eta', xticks=En_bins, yticks=Eta_bins)
        self.colormap(efficiency_En_phi, pdf_name='efficiency_EnPhi2D.png', xlabel='En', ylabel='phi', xticks=En_bins, yticks=Phi_bins)
        self.colormap(efficiency_En_pt, pdf_name='efficiency_EnPt2D.png', xlabel='En', ylabel='pt [GeV]', xticks=En_bins, yticks=PT_bins)
        self.colormap(efficiency_Phi_eta, pdf_name='efficiency_PhiEta2D.png', xlabel='phi', ylabel='eta', xticks=Phi_bins, yticks=Eta_bins)
        
        self.colormap(efficiency_En_eta_PF, pdf_name='efficiency_EnEta2D_PF.png', xlabel='En', ylabel='eta', xticks=En_bins, yticks=Eta_bins)
        self.colormap(efficiency_En_phi_PF, pdf_name='efficiency_EnPhi2D_PF.png', xlabel='En', ylabel='phi', xticks=En_bins, yticks=Phi_bins)
        self.colormap(efficiency_En_pt_PF, pdf_name='efficiency_EnPt2D_PF.png', xlabel='En', ylabel='pt [GeV]', xticks=En_bins, yticks=PT_bins)
        self.colormap(efficiency_Phi_eta_PF, pdf_name='efficiency_PhiEta2D_PF.png', xlabel='phi', ylabel='eta', xticks=Phi_bins, yticks=Eta_bins)
        


    def count(cp_dict):
        num_eta, denom_eta = [], []
        num_phi, denom_phi = [], []
        num_energy, denom_energy = [], []

        cumulative_eta, cumulative_phi, cumulative_energy = [], [], []

        # I work with the events where # calo particles > 0
        mask_moreThan0Calo = ak.count(cp_dict['caloParticleEnergy'], axis=1) > 0
        # I work with the events where # CP2LCscore > 0
        mask_moreThan0_CP2LCscore = ak.count(cp_dict['simToRecoAssociation'], axis=2) > 0

        denom_eta = cp_dict['caloParticleEta'][mask_moreThan0_CP2LCscore]
        denom_phi = cp_dict['caloParticlePhi'][mask_moreThan0_CP2LCscore]
        denom_energy = cp_dict['caloParticleEnergy'][mask_moreThan0_CP2LCscore]

        # I work with the events where # SharedEnergy > 0

        max_sharedEnergy = ak.max(cp_dict['sharedEnergy'], axis=2)
        max_sharedEnergy = max_sharedEnergy / cp_dict['caloParticleEnergy']

        maxSharedEnergy_largerThan07 = max_sharedEnergy > 0.7
        num_eta = cp_dict['caloParticleEta'][maxSharedEnergy_largerThan07]
        num_phi = cp_dict['caloParticlePhi'][maxSharedEnergy_largerThan07]
        num_energy = cp_dict['caloParticleEnergy'][maxSharedEnergy_largerThan07]

        # old code
        if False:

            for event in range(0, len(cp_dict)):
                event = str(event+1)
                # iterate over calo particles
                for cp in range(0, len(cp_dict[event]['caloParticleEnergy'])):
                    is_efficient = False
                    # if len of CP2LCscore for this event and for this calo particle == 0  => continue
                    if (len(cp_dict[event]['simToRecoAssociation'][cp]) == 0) or is_efficient : continue
                    denom_eta.append(cp_dict[event]['caloParticleEta'][cp])
                    denom_phi.append(cp_dict[event]['caloParticlePhi'][cp])
                    denom_energy.append(cp_dict[event]['caloParticleEnergy'][cp])

                    # if len of SharedEnergy for this event and for this calo particle == 0  => continue
                    if len(cp_dict[event]['sharedEnergy'][cp]) == 0 : continue
                    max_shared_energy = max(cp_dict[event]['sharedEnergy'][cp])
                    max_shared_energy /= cp_dict[event]['caloParticleEnergy'][cp]


                    if max_shared_energy > 0.7:
                        num_eta.append(cp_dict[event]['caloParticleEta'][cp])         
                        num_phi.append(cp_dict[event]['caloParticlePhi'][cp])
                        num_energy.append(cp_dict[event]['caloParticleEnergy'][cp])

        return [num_eta, denom_eta], [num_phi, denom_phi], [num_energy, denom_energy]


    def countCumulative(cp_dict):
        num_eta, denom_eta = [], []
        num_phi, denom_phi = [], []
        num_energy, denom_energy = [], []

        # I work with the events where # calo particles > 0
        mask_moreThan0Calo = ak.count(cp_dict['caloParticleEnergy'], axis=1) > 0
        # I work with the events where # CP2LCscore > 0
        mask_moreThan0_CP2LCscore = ak.count(cp_dict['simToRecoAssociation'], axis=2) > 0

        denom_eta = cp_dict['caloParticleEta'][mask_moreThan0_CP2LCscore]
        denom_phi = cp_dict['caloParticlePhi'][mask_moreThan0_CP2LCscore]
        denom_energy = cp_dict['caloParticleEnergy'][mask_moreThan0_CP2LCscore]

        # I work with the events where # SharedEnergy > 0

        total_shared_energy = ak.sum(cp_dict['sharedEnergy'], axis=2)
        total_shared_energy = total_shared_energy / cp_dict['caloParticleEnergy']

        maxSharedEnergy_largerThan07 = total_shared_energy > 0.7
        num_eta = cp_dict['caloParticleEta'][maxSharedEnergy_largerThan07]
        num_phi = cp_dict['caloParticlePhi'][maxSharedEnergy_largerThan07]
        num_energy = cp_dict['caloParticleEnergy'][maxSharedEnergy_largerThan07]

        # old code
        if False:
            for event in range(0, len(cp_dict)):
                event = str(event+1)
                for cp in range(0, len(cp_dict[event]['caloParticleEnergy'])):
                    total_shared_energy = 0 
                    is_efficient = False
                    if (len(cp_dict[event]['simToRecoAssociation'][cp]) == 0) or is_efficient : continue
                    denom_eta.append(cp_dict[event]['caloParticleEta'][cp])
                    denom_phi.append(cp_dict[event]['caloParticlePhi'][cp])
                    denom_energy.append(cp_dict[event]['caloParticleEnergy'][cp])
                    if len(cp_dict[event]['sharedEnergy'][cp]) == 0 : continue
                    for energy in cp_dict[event]['sharedEnergy'][cp]:
                        total_shared_energy += energy
                    total_shared_energy /= cp_dict[event]['caloParticleEnergy'][cp]
                    if total_shared_energy > 0.7:
                        is_efficient = True
                        num_eta.append(cp_dict[event]['caloParticleEta'][cp])
                        num_phi.append(cp_dict[event]['caloParticlePhi'][cp])
                        num_energy.append(cp_dict[event]['caloParticleEnergy'][cp])

        return [num_eta, denom_eta], [num_phi, denom_phi], [num_energy, denom_energy]

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
        
    def plot_bar_efficiency(self, efficiency, efficiency_err, bins, xlabel='blabla', title='', pdf_name='blabla', ylim=None, xlim=None, logY=False,
                           efficiency_pf=None, efficiency_err_pf=None):
        
        bins_avg = np.convolve(bins, [0.5, 0.5], "valid")
            
        #p1 = plt.scatter(bins_avg, efficiency, label='LC')
        p1 = plt.errorbar(bins_avg, efficiency, yerr=efficiency_err, fmt="o", color="r", label='LC')
        if efficiency_pf != None:
            #p1 = plt.scatter(bins_avg, efficiency_pf, label='PF')
            p1 = plt.errorbar(bins_avg, efficiency_pf, yerr=efficiency_err_pf, fmt="o", color="b", label='PF')

        plt.xlabel(xlabel)
        plt.ylabel('Efficiency')
        plt.ylim(ylim)
        plt.xlim(xlim)
        hep.cms.text(text=configuration["cmsText"])
        if logY:
            axs._setyscale('log')
            
        if efficiency_pf != None:
            plt.legend()
            
        plt.title(title, loc='right', fontsize=12)
        if (self.save != None):
            plt.savefig(self.save+pdf_name)

        plt.clf()
        
    def colormap(self, efficiency_2D, pdf_name='a.png', ylabel=None, xlabel=None, xticks=None, yticks=None, title=''):
        cmap = plt.cm.seismic

        heatmap = plt.pcolor(efficiency_2D, cmap=cmap, vmin=0.0, vmax=1.0)

        #legend
        cbar = plt.colorbar(heatmap)
        
        #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        # put the major ticks at the middle of each cell
        plt.xticks(np.arange(efficiency_2D.shape[1]) + 0.5, minor=False, labels=np.round(xticks[:-1], 2))
        plt.yticks(np.arange(efficiency_2D.shape[0]) + 0.5, minor=False, labels=np.round(yticks[:-1], 2))
        
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        
        hep.cms.text(text=configuration["cmsText"])
        plt.title(title, loc='right', fontsize=12)
        
        if (self.save != None):
            plt.savefig(self.save+pdf_name)

        plt.tight_layout()
        plt.clf()
        #plt.show()
    
    