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

class EfficiencyPlots_comparison:
    def __init__ (self, 
                  lc_dict_arr, cp_dict_arr,
                  cut_sim2Reco=[1.0],
                  compare_kappa=[0,1],
                  cut_sharedEn=[0.05,0.1,0.25,0.5,0.7],
                  label_kappa=['file_0', 'file_1'],
                  annotate=None,
                  legend=None,
                  c=['red', 'black'],
                  save=None):

        self.annotate=annotate
        self.legend=legend
        self.c = c
        self.lc_dict_arr = lc_dict_arr
        self.cp_dict_arr = cp_dict_arr
                
        self.label_kappa = label_kappa
        self.compare_kappa = compare_kappa
        
        self.En_bins = np.linspace(0, 50, 11)
        self.Eta_bins = np.linspace(-1.5, 1.5, 11)
        self.Phi_bins = np.linspace(-np.pi, np.pi, 11)
        self.PT_bins = np.linspace(0, 50, 11)
        
            
        # iterate over all thresholds
        for cut_sim2Reco_elem in cut_sim2Reco:

            self.sim2Reco_cut = cut_sim2Reco_elem

            self.save = save + f'/Comparison_kappa_efficiency_cutSim2Reco:{self.sim2Reco_cut}/'

            # Create the directory if doesn't exist
            if not os.path.exists(self.save):
                # Create the directory
                os.makedirs(self.save)            

            # plot results for current root file
            efficiency_forEachKappa_En, err_forEachKappa_En, \
            efficiency_forEachKappa_Eta, err_forEachKappa_Eta, \
            efficiency_forEachKappa_Phi, err_forEachKappa_Phi, \
            efficiency_forEachKappa_PT, err_forEachKappa_PT = self.computeEfficiency()
            
            self.plot_bar_efficiency_comparison(efficiency_forEachKappa_En, err_forEachKappa_En, self.En_bins, xlabel='En [GeV]', title='', pdf_name='efficiency_comp_En.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
            self.plot_bar_efficiency_comparison(efficiency_forEachKappa_Eta, err_forEachKappa_Eta, self.Eta_bins, xlabel='Eta', title='', pdf_name='efficiency_comp_Eta.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
            self.plot_bar_efficiency_comparison(efficiency_forEachKappa_Phi, err_forEachKappa_Phi, self.Phi_bins, xlabel='Phi', title='', pdf_name='efficiency_comp_Phi.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
            self.plot_bar_efficiency_comparison(efficiency_forEachKappa_PT, err_forEachKappa_PT, self.PT_bins, xlabel='pt [GeV]', title='', pdf_name='efficiency_comp_PT.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)


            for cut_sharedEn_elem in cut_sharedEn:

                self.sharedEn_cut = cut_sharedEn_elem

                self.save = save + f'/Comparison_kappa_efficiency_cutSim2Reco:{self.sim2Reco_cut}/sharedEn_{self.sharedEn_cut}/'

                # Create the directory if doesn't exist
                if not os.path.exists(self.save):
                    # Create the directory
                    os.makedirs(self.save)

                # cut on each individual LC
                efficiency_forEachKappa_En_cutShared, err_forEachKappa_En_cutShared, \
                efficiency_forEachKappa_Eta_cutShared, err_forEachKappa_Eta_cutShared, \
                efficiency_forEachKappa_Phi_cutShared, err_forEachKappa_Phi_cutShared, \
                efficiency_forEachKappa_PT_cutShared, err_forEachKappa_PT_cutShared = self.computeEfficiency_sharedEnCut()

                self.plot_bar_efficiency_comparison(efficiency_forEachKappa_En_cutShared, err_forEachKappa_En_cutShared, self.En_bins, xlabel='En [GeV]', title='', pdf_name='efficiency_comp_En.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
                self.plot_bar_efficiency_comparison(efficiency_forEachKappa_Eta_cutShared, err_forEachKappa_Eta_cutShared, self.Eta_bins, xlabel='Eta', title='', pdf_name='efficiency_comp_Eta.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
                self.plot_bar_efficiency_comparison(efficiency_forEachKappa_Phi_cutShared, err_forEachKappa_Phi_cutShared, self.Phi_bins, xlabel='Phi', title='', pdf_name='efficiency_comp_Phi.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
                self.plot_bar_efficiency_comparison(efficiency_forEachKappa_PT_cutShared, err_forEachKappa_PT_cutShared, self.PT_bins, xlabel='pt [GeV]', title='', pdf_name='efficiency_comp_PT.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)


                # cut on each individual LC
                efficiency_forEachKappa_En_cutShared_sum, err_forEachKappa_En_cutShared_sum, \
                efficiency_forEachKappa_Eta_cutShared_sum, err_forEachKappa_Eta_cutShared_sum, \
                efficiency_forEachKappa_Phi_cutShared_sum, err_forEachKappa_Phi_cutShared_sum, \
                efficiency_forEachKappa_PT_cutShared_sum, err_forEachKappa_PT_cutShared_sum = self.computeEfficiency_sharedEnCut_sum()

                self.plot_bar_efficiency_comparison(efficiency_forEachKappa_En_cutShared_sum, err_forEachKappa_En_cutShared_sum, self.En_bins, xlabel='En [GeV]', title='', pdf_name='efficiency_comp_En_sum.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
                self.plot_bar_efficiency_comparison(efficiency_forEachKappa_Eta_cutShared_sum, err_forEachKappa_Eta_cutShared_sum, self.Eta_bins, xlabel='Eta', title='', pdf_name='efficiency_comp_Eta_sum.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
                self.plot_bar_efficiency_comparison(efficiency_forEachKappa_Phi_cutShared_sum, err_forEachKappa_Phi_cutShared_sum, self.Phi_bins, xlabel='Phi', title='', pdf_name='efficiency_comp_Phi_sum.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
                self.plot_bar_efficiency_comparison(efficiency_forEachKappa_PT_cutShared_sum, err_forEachKappa_PT_cutShared_sum, self.PT_bins, xlabel='pt [GeV]', title='', pdf_name='efficiency_comp_PT_sum.png', ylim=(0,1.05), logY=False, label_kappa=self.label_kappa)
                
            
            
            
        
    def compute_eff_perBin(self, caloP_kinematics, noAssociatedLC_perEv, bins, debug=False):
        
        efficiencyPerBin = [-1] * (len(bins) - 1)
        errPerBin = [-1] * (len(bins) - 1)
        
        for i in range(len(bins) - 1):
            mask_lower = caloP_kinematics <= bins[i+1]
            mask_higher = caloP_kinematics > bins[i]
            
            mask_Events = np.logical_and(mask_lower.to_numpy(), mask_higher.to_numpy())
                        
            mask_AssociatedLC = noAssociatedLC_perEv[mask_Events] > 0
            
            if debug:
                print(noAssociatedLC_perEv)
                print(mask_Events)

                print(ak.num(mask_Events, axis=0))
                print(ak.count_nonzero(mask_Events))
                print(mask_AssociatedLC)
                print(ak.num(mask_AssociatedLC, axis=0))
                print(ak.count_nonzero(mask_AssociatedLC))
                print()
            
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
            
    def computeEfficiency(self):
        # efficiency for each kappa + for PF
        efficiency_forEachKappa_En = []
        efficiency_forEachKappa_Eta = []
        efficiency_forEachKappa_Phi = []
        efficiency_forEachKappa_PT = []
        err_forEachKappa_En = []
        err_forEachKappa_Eta = []
        err_forEachKappa_Phi = []
        err_forEachKappa_PT = []
                
        for i in range(len(self.cp_dict_arr)):
                        
            curr_arr = self.cp_dict_arr[i]
            
            # when is not part of comparison and it's not the last one (PF)
            if i not in self.compare_kappa and i < len(self.cp_dict_arr) - 1:
                continue
                
            mask_sim2Reco_cut = curr_arr['simToRecoAssociation'] <= self.sim2Reco_cut
            noAssociatedLC_perEv = ak.num(curr_arr['AssociatedLC'][mask_sim2Reco_cut], axis=2)
            
            curr_arr_pt = curr_arr['caloParticleEnergy'].to_numpy()/np.cosh(curr_arr['caloParticleEta'].to_numpy())
        
            efficiency_forEachKappa_En_curr, err_forEachKappa_En_curr = self.compute_eff_perBin(curr_arr['caloParticleEnergy'], noAssociatedLC_perEv, self.En_bins)
            efficiency_forEachKappa_En.append(efficiency_forEachKappa_En_curr)
            err_forEachKappa_En.append(err_forEachKappa_En_curr)
            
            efficiency_forEachKappa_Eta_curr, err_forEachKappa_Eta_curr = self.compute_eff_perBin(curr_arr['caloParticleEta'], noAssociatedLC_perEv, self.Eta_bins)
            efficiency_forEachKappa_Eta.append(efficiency_forEachKappa_Eta_curr)
            err_forEachKappa_Eta.append(err_forEachKappa_Eta_curr)
            
            
            efficiency_forEachKappa_Phi_curr, err_forEachKappa_Phi_curr = self.compute_eff_perBin(curr_arr['caloParticlePhi'], noAssociatedLC_perEv, self.Phi_bins)
            efficiency_forEachKappa_Phi.append(efficiency_forEachKappa_Phi_curr)
            err_forEachKappa_Phi.append(err_forEachKappa_Phi_curr)
            
            efficiency_forEachKappa_PT_curr, err_forEachKappa_PT_curr = self.compute_eff_perBin(ak.Array(curr_arr_pt), noAssociatedLC_perEv, self.PT_bins)
            efficiency_forEachKappa_PT.append(efficiency_forEachKappa_PT_curr)
            err_forEachKappa_PT.append(err_forEachKappa_PT_curr)
            
            maskIneff_Ev = noAssociatedLC_perEv == 0
            
            self.hist_2d(ak.flatten(curr_arr['caloParticleEta'][maskIneff_Ev]).to_numpy(),
                         ak.flatten(curr_arr['caloParticlePhi'][maskIneff_Ev]).to_numpy(),
                         xlabel='caloP ETA',
                ylabel='caloP PHI', title=f'caloP Ineff: cut sim2Reco = {self.sim2Reco_cut}',
                pdf_name=f'caloP_eta_phi_ineff.png', logY=False, binning=(self.Eta_bins, self.Phi_bins))
            
        return efficiency_forEachKappa_En, err_forEachKappa_En, \
                efficiency_forEachKappa_Eta, err_forEachKappa_Eta, \
                efficiency_forEachKappa_Phi, err_forEachKappa_Phi, \
                efficiency_forEachKappa_PT, err_forEachKappa_PT


    def computeEfficiency_sharedEnCut(self):
        # efficiency for each kappa + for PF
        efficiency_forEachKappa_En = []
        efficiency_forEachKappa_Eta = []
        efficiency_forEachKappa_Phi = []
        efficiency_forEachKappa_PT = []
        err_forEachKappa_En = []
        err_forEachKappa_Eta = []
        err_forEachKappa_Phi = []
        err_forEachKappa_PT = []
                
        for i in range(len(self.cp_dict_arr)):
            
            debug = False
            if self.sharedEn_cut == 0.7:
                debug=True
                        
            curr_arr = self.cp_dict_arr[i]
            curr_arr_sharedEn_percentage = curr_arr['sharedEnergy']/curr_arr['caloParticleEnergy']
            
            # when is not part of comparison and it's not the last one (PF)
            if i not in self.compare_kappa and i < len(self.cp_dict_arr) - 1:
                continue
                
            mask_sim2Reco_cut = curr_arr['simToRecoAssociation'] <= self.sim2Reco_cut
            mask_sharedEn_cut = curr_arr_sharedEn_percentage[mask_sim2Reco_cut] >= self.sharedEn_cut

            print(f"SHARED EN ---------------------------------------------------- {self.sharedEn_cut} & i = {i}")
            if debug:
                print(curr_arr_sharedEn_percentage)
                print(mask_sharedEn_cut)
                print(ak.num(mask_sharedEn_cut, axis=0))
                print(ak.num(mask_sharedEn_cut))
                print(ak.count_nonzero(mask_sharedEn_cut))
                x = ak.any(mask_sharedEn_cut, axis=2)
                print(ak.count_nonzero(x))
                print()
            
            noAssociatedLC_perEv = ak.num(curr_arr['AssociatedLC'][mask_sharedEn_cut], axis=2)

            if debug:
                print(curr_arr['AssociatedLC'])
                print(curr_arr['AssociatedLC'][mask_sharedEn_cut])
                print(ak.num(curr_arr['AssociatedLC'][mask_sharedEn_cut], axis=2))
                x = ak.num(curr_arr['AssociatedLC'][mask_sharedEn_cut], axis=2) > 0
                print(x)
                print(ak.count_nonzero(x))
                print('finishhh \n\n\n')
            #print(noAssociatedLC_perEv)
            #print()
            
            curr_arr_pt = curr_arr['caloParticleEnergy'].to_numpy()/np.cosh(curr_arr['caloParticleEta'].to_numpy())
        
            efficiency_forEachKappa_En_curr, err_forEachKappa_En_curr = self.compute_eff_perBin(curr_arr['caloParticleEnergy'], noAssociatedLC_perEv, self.En_bins, debug=debug)
            efficiency_forEachKappa_En.append(efficiency_forEachKappa_En_curr)
            err_forEachKappa_En.append(err_forEachKappa_En_curr)
            
            efficiency_forEachKappa_Eta_curr, err_forEachKappa_Eta_curr = self.compute_eff_perBin(curr_arr['caloParticleEta'], noAssociatedLC_perEv, self.Eta_bins)
            efficiency_forEachKappa_Eta.append(efficiency_forEachKappa_Eta_curr)
            err_forEachKappa_Eta.append(err_forEachKappa_Eta_curr)
            
            
            efficiency_forEachKappa_Phi_curr, err_forEachKappa_Phi_curr = self.compute_eff_perBin(curr_arr['caloParticlePhi'], noAssociatedLC_perEv, self.Phi_bins)
            efficiency_forEachKappa_Phi.append(efficiency_forEachKappa_Phi_curr)
            err_forEachKappa_Phi.append(err_forEachKappa_Phi_curr)
            
            efficiency_forEachKappa_PT_curr, err_forEachKappa_PT_curr = self.compute_eff_perBin(ak.Array(curr_arr_pt), noAssociatedLC_perEv, self.PT_bins)
            efficiency_forEachKappa_PT.append(efficiency_forEachKappa_PT_curr)
            err_forEachKappa_PT.append(err_forEachKappa_PT_curr)
            
            maskIneff_Ev = noAssociatedLC_perEv == 0
            
            self.hist_2d(ak.flatten(curr_arr['caloParticleEta'][maskIneff_Ev]).to_numpy(),
                         ak.flatten(curr_arr['caloParticlePhi'][maskIneff_Ev]).to_numpy(),
                         xlabel='caloP ETA',
                ylabel='caloP PHI', title=f'caloP Ineff: cut sharedEn = {self.sharedEn_cut}',
                pdf_name=f'caloP_eta_phi_ineff.png', logY=False, binning=(self.Eta_bins, self.Phi_bins))
            
        return efficiency_forEachKappa_En, err_forEachKappa_En, \
                efficiency_forEachKappa_Eta, err_forEachKappa_Eta, \
                efficiency_forEachKappa_Phi, err_forEachKappa_Phi, \
                efficiency_forEachKappa_PT, err_forEachKappa_PT


    def computeEfficiency_sharedEnCut_sum(self):
        # efficiency for each kappa + for PF
        efficiency_forEachKappa_En = []
        efficiency_forEachKappa_Eta = []
        efficiency_forEachKappa_Phi = []
        efficiency_forEachKappa_PT = []
        err_forEachKappa_En = []
        err_forEachKappa_Eta = []
        err_forEachKappa_Phi = []
        err_forEachKappa_PT = []
                
        for i in range(len(self.cp_dict_arr)):
                        
            curr_arr = self.cp_dict_arr[i]
            
            # when is not part of comparison and it's not the last one (PF)
            if i not in self.compare_kappa and i < len(self.cp_dict_arr) - 1:
                continue
                
            mask_sim2Reco_cut = curr_arr['simToRecoAssociation'] <= self.sim2Reco_cut

            curr_arr_sharedEn_percentage = curr_arr['sharedEnergy'][mask_sim2Reco_cut]/curr_arr['caloParticleEnergy']
            curr_arr_sharedEn_percentage_sum = ak.sum(curr_arr_sharedEn_percentage, axis=2)
            
            mask_sharedEn_cut = curr_arr_sharedEn_percentage_sum >= self.sharedEn_cut
            
            noAssociatedLC_perEv = ak.count_nonzero(mask_sharedEn_cut, axis=1)
            noAssociatedLC_perEv = noAssociatedLC_perEv[:,np.newaxis]

            #print(noAssociatedLC_perEv)
            #print(ak.num(noAssociatedLC_perEv, axis=0))
            #print(ak.num(noAssociatedLC_perEv, axis=1))
            
            curr_arr_pt = curr_arr['caloParticleEnergy'].to_numpy()/np.cosh(curr_arr['caloParticleEta'].to_numpy())
        
            efficiency_forEachKappa_En_curr, err_forEachKappa_En_curr = self.compute_eff_perBin(curr_arr['caloParticleEnergy'], noAssociatedLC_perEv, self.En_bins)
            efficiency_forEachKappa_En.append(efficiency_forEachKappa_En_curr)
            err_forEachKappa_En.append(err_forEachKappa_En_curr)
            
            efficiency_forEachKappa_Eta_curr, err_forEachKappa_Eta_curr = self.compute_eff_perBin(curr_arr['caloParticleEta'], noAssociatedLC_perEv, self.Eta_bins)
            efficiency_forEachKappa_Eta.append(efficiency_forEachKappa_Eta_curr)
            err_forEachKappa_Eta.append(err_forEachKappa_Eta_curr)
            
            
            efficiency_forEachKappa_Phi_curr, err_forEachKappa_Phi_curr = self.compute_eff_perBin(curr_arr['caloParticlePhi'], noAssociatedLC_perEv, self.Phi_bins)
            efficiency_forEachKappa_Phi.append(efficiency_forEachKappa_Phi_curr)
            err_forEachKappa_Phi.append(err_forEachKappa_Phi_curr)
            
            efficiency_forEachKappa_PT_curr, err_forEachKappa_PT_curr = self.compute_eff_perBin(ak.Array(curr_arr_pt), noAssociatedLC_perEv, self.PT_bins)
            efficiency_forEachKappa_PT.append(efficiency_forEachKappa_PT_curr)
            err_forEachKappa_PT.append(err_forEachKappa_PT_curr)
            
            maskIneff_Ev = noAssociatedLC_perEv == 0
            
            self.hist_2d(curr_arr['caloParticleEta'][maskIneff_Ev].to_numpy(),
                         curr_arr['caloParticlePhi'][maskIneff_Ev].to_numpy(),
                         xlabel='caloP ETA',
                ylabel='caloP PHI', title=f'caloP Ineff: cut sharedEn = {self.sharedEn_cut}',
                pdf_name=f'caloP_eta_phi_ineff_sum.png', logY=False, binning=(self.Eta_bins, self.Phi_bins))
            
        return efficiency_forEachKappa_En, err_forEachKappa_En, \
                efficiency_forEachKappa_Eta, err_forEachKappa_Eta, \
                efficiency_forEachKappa_Phi, err_forEachKappa_Phi, \
                efficiency_forEachKappa_PT, err_forEachKappa_PT
                
     
    def plot_bar_efficiency_comparison(self, efficiency, efficiency_err, bins, xlabel='blabla', title='', pdf_name='blabla', ylim=None, xlim=None, logY=False, label_kappa=None):
        
        bins_avg = np.convolve(bins, [0.5, 0.5], "valid")

        color=['red', 'green', 'blue', 'cyan', 'magenta', 'black', 'purple']
            
        for i in range(len(self.cp_dict_arr)):
                        
            if i == len(self.cp_dict_arr) - 1:
                p1 = plt.errorbar(bins_avg, efficiency[i], yerr=efficiency_err[i], color=color[i], fmt="o", label='PF')
        
            elif i in self.compare_kappa:
                p1 = plt.errorbar(bins_avg, efficiency[i], yerr=efficiency_err[i], color=color[i], fmt="o", label=label_kappa[i])
       
        plt.xlabel(xlabel)
        plt.ylabel('Efficiency')
        plt.ylim(ylim)
        plt.xlim(xlim)
        hep.cms.text(text=configuration["cmsText"])
        if logY:
            axs._setyscale('log')
            
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
    
    def hist_2d(self, arr1, arr2, xlabel='blabla', ylabel='blabla', title='', pdf_name='blabla', logY=False, binning=None, annotate=None):
        fig, ax = plt.subplots()

        h = ax.hist2d(arr1, arr2, bins=binning, cmin=1)
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.savefig(self.save+pdf_name)
        plt.close(fig)
    