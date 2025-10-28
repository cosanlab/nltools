"""WIP"""
# class Model(object):
#     def __init__(self):
#         pass

#     def predict(self, algorithm=None, cv_dict=None, plot=True, verbose=True, **kwargs):
#         """Run prediction

#         Args:
#             algorithm: Algorithm to use for prediction.  Must be one of 'svm',
#                     'svr', 'linear', 'logistic', 'lasso', 'ridge',
#                     'ridgeClassifier','pcr', or 'lassopcr'
#             cv_dict: Type of cross_validation to use. A dictionary of
#                     {'type': 'kfolds', 'n_folds': n},
#                     {'type': 'kfolds', 'n_folds': n, 'stratified': Y},
#                     {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
#                     {'type': 'loso', 'subject_id': holdout}
#                     where 'n' = number of folds, and 'holdout' = vector of
#                     subject ids that corresponds to self.Y
#             plot: Boolean indicating whether or not to create plots.
#             verbose (bool): print performance; Default True
#             **kwargs: Additional keyword arguments to pass to the prediction
#                     algorithm

#         Returns:
#             output: a dictionary of prediction parameters

#         """

#         # Set algorithm
#         if algorithm is not None:
#             predictor_settings = set_algorithm(algorithm, **kwargs)
#         else:
#             # Use SVR as a default
#             predictor_settings = set_algorithm("svr", **{"kernel": "linear"})

#         # Initialize output dictionary
#         output = {"Y": np.array(self.Y).flatten()}
#         predictor = predictor_settings["predictor"]

#         # Overall Fit for weight map
#         predictor.fit(self.data, np.ravel(output["Y"]))
#         output["yfit_all"] = predictor.predict(self.data)
#         if predictor_settings["prediction_type"] == "classification":
#             if predictor_settings["algorithm"] not in [
#                 "svm",
#                 "ridgeClassifier",
#                 "ridgeClassifierCV",
#             ]:
#                 output["prob_all"] = predictor.predict_proba(self.data)
#             else:
#                 output["dist_from_hyperplane_all"] = predictor.decision_function(
#                     self.data
#                 )
#                 if predictor_settings["algorithm"] == "svm" and predictor.probability:
#                     output["prob_all"] = predictor.predict_proba(self.data)

#         # Intercept
#         if predictor_settings["algorithm"] == "pcr":
#             output["intercept"] = predictor_settings["_regress"].intercept_
#         elif predictor_settings["algorithm"] == "lassopcr":
#             output["intercept"] = predictor_settings["_lasso"].intercept_
#         else:
#             output["intercept"] = predictor.intercept_

#         # Weight map
#         output["weight_map"] = self.empty()
#         if predictor_settings["algorithm"] == "lassopcr":
#             output["weight_map"].data = np.dot(
#                 predictor_settings["_pca"].components_.T,
#                 predictor_settings["_lasso"].coef_,
#             )
#         elif predictor_settings["algorithm"] == "pcr":
#             output["weight_map"].data = np.dot(
#                 predictor_settings["_pca"].components_.T,
#                 predictor_settings["_regress"].coef_,
#             )
#         else:
#             output["weight_map"].data = predictor.coef_.squeeze()

#         # Cross-Validation Fit
#         from sklearn.base import clone

#         if cv_dict is not None:
#             cv = set_cv(Y=self.Y, cv_dict=cv_dict)

#             predictor_cv = predictor_settings["predictor"]
#             output["yfit_xval"] = output["yfit_all"].copy()
#             output["intercept_xval"] = []
#             # Multi-class classification, init weightmaps as list
#             if (predictor_settings["prediction_type"] == "classification") and (
#                 len(np.unique(self.Y)) > 2
#             ):
#                 output["weight_map_xval"] = []
#             else:
#                 # Otherwise we'll have a single weightmap
#                 output["weight_map_xval"] = output["weight_map"].copy()
#             output["cv_idx"] = []
#             wt_map_xval = []

#             # Initialize zero'd arrays that will be filled during cross-validation and fitting
#             # These will need change shape if doing multi-class or probablistic predictions
#             if (predictor_settings["algorithm"] == "logistic") or (
#                 predictor_settings["algorithm"] == "svm" and predictor.probability
#             ):
#                 # If logistic or svm prob, probs == number of classes
#                 probs_init = np.zeros((len(self.Y), len(np.unique(self.Y))))
#             # however if num classes == 2 decision function == 1, but if num class > 2, decision function == num classes (sklearn weirdness)
#             if len(np.unique(self.Y)) == 2:
#                 dec_init = np.zeros(len(self.Y))
#             else:
#                 dec_init = np.zeros((len(self.Y), len(np.unique(self.Y))))
#             # else:
#             #
#             #     if len(np.unique(self.Y)) == 2:
#             #         dec_init = np.zeros(len(self.Y))
#             #     else:
#             #         dec_init = np.zeros((len(self.Y), len(np.unique(self.Y))))

#             if predictor_settings["prediction_type"] == "classification":
#                 if predictor_settings["algorithm"] not in [
#                     "svm",
#                     "ridgeClassifier",
#                     "ridgeClassifierCV",
#                 ]:
#                     output["prob_xval"] = probs_init
#                 else:
#                     output["dist_from_hyperplane_xval"] = dec_init
#                     if (
#                         predictor_settings["algorithm"] == "svm"
#                         and predictor_cv.probability
#                     ):
#                         output["prob_xval"] = probs_init

#             for train, test in cv:
#                 # Ensure estimators are always indepedent across folds
#                 predictor_cv = clone(predictor_settings["predictor"])
#                 predictor_cv.fit(self.data[train], np.ravel(self.Y.iloc[train]))
#                 output["yfit_xval"][test] = predictor_cv.predict(
#                     self.data[test]
#                 ).ravel()
#                 if predictor_settings["prediction_type"] == "classification":
#                     if predictor_settings["algorithm"] not in [
#                         "svm",
#                         "ridgeClassifier",
#                         "ridgeClassifierCV",
#                     ]:
#                         output["prob_xval"][test] = predictor_cv.predict_proba(
#                             self.data[test]
#                         )
#                     else:
#                         output["dist_from_hyperplane_xval"][test] = (
#                             predictor_cv.decision_function(self.data[test])
#                         )
#                         if (
#                             predictor_settings["algorithm"] == "svm"
#                             and predictor_cv.probability
#                         ):
#                             output["prob_xval"][test] = predictor_cv.predict_proba(
#                                 self.data[test]
#                             )
#                 # Intercept
#                 if predictor_settings["algorithm"] == "pcr":
#                     output["intercept_xval"].append(predictor_cv["regress"].intercept_)
#                 elif predictor_settings["algorithm"] == "lassopcr":
#                     output["intercept_xval"].append(predictor_cv["lasso"].intercept_)
#                 else:
#                     output["intercept_xval"].append(predictor_cv.intercept_)
#                 output["cv_idx"].append((train, test))

#                 # Weight map
#                 # Multi-class classification, weightmaps as list
#                 if (predictor_settings["prediction_type"] == "classification") and (
#                     len(np.unique(self.Y)) > 2
#                 ):
#                     tmp = output["weight_map"].empty()
#                     tmp.data = predictor_cv.coef_.squeeze()
#                     output["weight_map_xval"].append(tmp)
#                 # Regression or binary classification
#                 else:
#                     if predictor_settings["algorithm"] == "lassopcr":
#                         wt_map_xval.append(
#                             np.dot(
#                                 predictor_cv["pca"].components_.T,
#                                 predictor_cv["lasso"].coef_,
#                             )
#                         )
#                     elif predictor_settings["algorithm"] == "pcr":
#                         wt_map_xval.append(
#                             np.dot(
#                                 predictor_cv["pca"].components_.T,
#                                 predictor_cv["regress"].coef_,
#                             )
#                         )
#                     else:
#                         wt_map_xval.append(predictor_cv.coef_.squeeze())
#                     output["weight_map_xval"].data = np.array(wt_map_xval)

#         # Print Results
#         if predictor_settings["prediction_type"] == "classification":
#             output["mcr_all"] = balanced_accuracy_score(
#                 self.Y.values, output["yfit_all"]
#             )
#             if verbose:
#                 print("overall accuracy: %.2f" % output["mcr_all"])
#             if cv_dict is not None:
#                 output["mcr_xval"] = np.mean(
#                     output["yfit_xval"] == np.array(self.Y).flatten()
#                 )
#                 if verbose:
#                     print("overall CV accuracy: %.2f" % output["mcr_xval"])
#         elif predictor_settings["prediction_type"] == "prediction":
#             output["rmse_all"] = np.sqrt(
#                 np.mean((output["yfit_all"] - output["Y"]) ** 2)
#             )
#             output["r_all"] = pearsonr(output["Y"], output["yfit_all"])[0]
#             if verbose:
#                 print("overall Root Mean Squared Error: %.2f" % output["rmse_all"])
#                 print("overall Correlation: %.2f" % output["r_all"])
#             if cv_dict is not None:
#                 output["rmse_xval"] = np.sqrt(
#                     np.mean((output["yfit_xval"] - output["Y"]) ** 2)
#                 )
#                 output["r_xval"] = pearsonr(output["Y"], output["yfit_xval"])[0]
#                 if verbose:
#                     print(
#                         "overall CV Root Mean Squared Error: %.2f" % output["rmse_xval"]
#                     )
#                     print("overall CV Correlation: %.2f" % output["r_xval"])

#         # Plot
#         if plot:
#             if cv_dict is not None:
#                 if predictor_settings["prediction_type"] == "prediction":
#                     scatterplot(
#                         pd.DataFrame(
#                             {"Y": output["Y"], "yfit_xval": output["yfit_xval"]}
#                         )
#                     )
#                 elif predictor_settings["prediction_type"] == "classification":
#                     if len(np.unique(self.Y)) > 2:
#                         print("Skipping ROC plot because num_classes > 2")
#                     else:
#                         if predictor_settings["algorithm"] not in [
#                             "svm",
#                             "ridgeClassifier",
#                             "ridgeClassifierCV",
#                         ]:
#                             output["roc"] = Roc(
#                                 input_values=output["prob_xval"][:, 1],
#                                 binary_outcome=output["Y"].astype("bool"),
#                             )
#                         else:
#                             output["roc"] = Roc(
#                                 input_values=output["dist_from_hyperplane_xval"],
#                                 binary_outcome=output["Y"].astype("bool"),
#                             )
#                             if (
#                                 predictor_settings["algorithm"] == "svm"
#                                 and predictor_cv.probability
#                             ):
#                                 output["roc"] = Roc(
#                                     input_values=output["prob_xval"][:, 1],
#                                     binary_outcome=output["Y"].astype("bool"),
#                                 )
#                         output["roc"].plot()
#             output["weight_map"].plot()

#         return output

#     def predict_multi(
#         self,
#         algorithm=None,
#         cv_dict=None,
#         method="searchlight",
#         rois=None,
#         process_mask=None,
#         radius=2.0,
#         scoring=None,
#         n_jobs=1,
#         verbose=0,
#         **kwargs,
#     ):
#         """Perform multi-region prediction. This can be a searchlight analysis or multi-roi analysis if provided a Brain_Data instance with labeled non-overlapping rois.

#         Args:
#             algorithm (string): algorithm to use for prediction Must be one of 'svm',
#                     'svr', 'linear', 'logistic', 'lasso', 'ridge',
#                     'ridgeClassifier','pcr', or 'lassopcr'
#             cv_dict: Type of cross_validation to use. Default is 3-fold. A dictionary of
#                     {'type': 'kfolds', 'n_folds': n},
#                     {'type': 'kfolds', 'n_folds': n, 'stratified': Y},
#                     {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
#                     {'type': 'loso', 'subject_id': holdout}
#                     where 'n' = number of folds, and 'holdout' = vector of
#                     subject ids that corresponds to self.Y
#             method (string): one of 'searchlight' or 'roi'
#             rois (string/nltools.Brain_Data): nifti file path or Brain_data instance containing non-overlapping regions-of-interest labeled by integers
#             process_mask (nib.Nifti1Image/nltools.Brain_Data): mask to constrain where to perform analyses; only applied if method = 'searchlight'
#             radius (float): radius of searchlight in mm; default 2mm
#             scoring (function): callable scoring function; see sklearn documentation; defaults to estimator's default scoring function
#             n_jobs (int): The number of CPUs to use to do permutation; default 1 because this can be very memory intensive
#             verbose (int): whether parallelization progress should be printed; default 0

#         Returns:
#             output: image of results

#         """

#         if method not in ["searchlight", "rois"]:
#             raise ValueError("method must be one of 'searchlight' or 'roi'")
#         if method == "roi" and rois is None:
#             raise ValueError(
#                 "With method = 'roi' a file path, or nibabel/nltools instance with roi labels must be provided"
#             )

#         # Set algorithm
#         if algorithm is not None:
#             predictor_settings = set_algorithm(algorithm, **kwargs)
#         else:
#             # Use SVR as a default
#             predictor_settings = set_algorithm("svr", **{"kernel": "linear"})
#         estimator = predictor_settings["predictor"]

#         if cv_dict is not None:
#             cv = set_cv(Y=self.Y, cv_dict=cv_dict, return_generator=False)
#             groups = cv_dict["subject_id"] if cv_dict["type"] == "loso" else None
#         else:
#             cv = None
#             groups = None

#         if method == "rois":
#             if isinstance(rois, str) or isinstance(rois, Path):
#                 if os.path.isfile(rois):
#                     rois_img = Brain_Data(rois, mask=self.mask)
#             elif isinstance(rois, Brain_Data):
#                 rois_img = rois.copy()
#             else:
#                 raise TypeError("rois must be a file path or a Brain_Data instance")
#             if len(rois_img.shape()) == 1:
#                 rois_img = expand_mask(rois_img, custom_mask=self.mask)
#             if len(rois_img.shape()) != 2:
#                 raise ValueError(
#                     "rois cannot be coerced into a mask. Make sure nifti file or Brain_Data is 3d with non-overlapping integer labels or 4d with non-overlapping boolean masks"
#                 )

#             out = Parallel(n_jobs=n_jobs, verbose=verbose)(
#                 delayed(_roi_func)(self, r, algorithm, cv_dict, **kwargs)
#                 for r in rois_img
#             )

#         elif method == "searchlight":
#             # Searchlight
#             if process_mask is None:
#                 process_mask_img = None
#             elif isinstance(process_mask, nib.Nifti1Image):
#                 process_mask_img = process_mask
#             elif isinstance(process_mask, Brain_Data):
#                 process_mask_img = process_mask.to_nifti()
#             elif isinstance(process_mask, str) or isinstance(process_mask, Path):
#                 if os.path.isfile(process_mask):
#                     process_mask_img = nib.load(process_mask)
#                 else:
#                     raise ValueError(
#                         "process mask file path specified but can't be found"
#                     )
#             else:
#                 raise TypeError(
#                     "process_mask is not a valid nibabel instance, Brain_Data instance or file path"
#                 )

#             sl = SearchLight(
#                 mask_img=self.mask,
#                 process_mask_img=process_mask_img,
#                 estimator=estimator,
#                 n_jobs=n_jobs,
#                 scoring=scoring,
#                 cv=cv,
#                 verbose=verbose,
#                 radius=radius,
#             )
#             in_image = self.to_nifti()
#             sl.fit(in_image, self.Y, groups=groups)
#             out = nib.Nifti1Image(sl.scores_, affine=self.nifti_masker.affine_)
#             out = Brain_Data(out, mask=self.mask)
#         return out

#     def ttest(self, threshold_dict=None, return_mask=False):
#         """Calculate one sample t-test across each voxel (two-sided)

#         Args:
#             threshold_dict: (dict) a dictionary of threshold parameters
#                             {'unc':.001} or {'fdr':.05}
#             return_mask: (bool) if thresholding is requested, optionall return the mask of voxels that exceed threshold, e.g. for use with another map

#         Returns:
#             out: (dict) dictionary of regression statistics in Brain_Data
#                  instances {'t','p'}

#         """

#         # TODO: remove copy
#         t = deepcopy(self)
#         # TODO: remove copy
#         p = deepcopy(self)

#         t.data, p.data = ttest_1samp(self.data, 0, 0)

#         if threshold_dict is not None:
#             if isinstance(threshold_dict, dict):
#                 if "unc" in threshold_dict:
#                     thr = threshold_dict["unc"]
#                 elif "fdr" in threshold_dict:
#                     thr = fdr(p.data, q=threshold_dict["fdr"])
#                 elif "holm-bonf" in threshold_dict:
#                     thr = holm_bonf(p.data, alpha=threshold_dict["holm-bonf"])

#                 if return_mask:
#                     thr_t, thr_mask = threshold(t, p, thr, True)
#                     out = {"t": t, "p": p, "thr_t": thr_t, "thr_mask": thr_mask}
#                 else:
#                     thr_t = threshold(t, p, thr)
#                     out = {"t": t, "p": p, "thr_t": thr_t}
#             else:
#                 raise ValueError(
#                     "threshold_dict is not a dictionary. "
#                     "Make sure it is in the form of {'unc': .001} "
#                     "or {'fdr': .05}"
#                 )
#         else:
#             out = {"t": t, "p": p}

#         return out

#     def randomise(
#         self, n_permute=5000, threshold_dict=None, return_mask=False, **kwargs
#     ):
#         """
#         Run mass-univariate regression at each voxel with inference performed
#         via permutation testing ala randomise in FSL. Operates just like
#         .regress(), but intended to be used for second-level analyses.

#         Args:
#             n_permute (int): number of permutations
#             threshold_dict: (dict) a dictionary of threshold parameters
#                             {'unc':.001} or {'fdr':.05}
#             return_mask: (bool) optionally return the thresholding mask
#         Returns:
#             out: dictionary of maps for betas, tstats, and pvalues
#         """

#         if not isinstance(self.X, pd.DataFrame):
#             raise ValueError("Make sure self.X is a pandas DataFrame.")

#         if self.X.empty:
#             raise ValueError("Make sure self.X is not empty.")

#         if self.data.shape[0] != self.X.shape[0]:
#             raise ValueError("self.X does not match the correct size of self.data")

#         b, t, p = regress_permutation(self.X, self.data, n_permute=n_permute, **kwargs)

#         # Prevent copy of all data in self multiple times; instead start with an empty instance and copy only needed attributes from self, and use this as a template for other outputs
#         b_out = self.__class__()
#         # TODO: remove copy
#         b_out.mask = deepcopy(self.mask)
#         # TODO: remove copy
#         b_out.nifti_masker = deepcopy(self.nifti_masker)

#         # Use this as template for other outputs before setting data
#         t_out = b_out.copy()
#         p_out = b_out.copy()
#         b_out.data, t_out.data, p_out.data = (b, t, p)

#         if threshold_dict is not None:
#             if isinstance(threshold_dict, dict):
#                 if "unc" in threshold_dict:
#                     thr = threshold_dict["unc"]
#                 elif "fdr" in threshold_dict:
#                     thr = fdr(p_out.data, q=threshold_dict["fdr"])
#                 elif "holm-bof" in threshold_dict:
#                     thr = holm_bonf(p.data, alpha=threshold_dict["holm-bonf"])
#                 elif "permutation" in threshold_dict:
#                     thr = 0.05
#                 if return_mask:
#                     thr_t_out, thr_mask = threshold(t_out, p_out, thr, True)
#                     out = {
#                         "beta": b_out,
#                         "t": t_out,
#                         "p": p_out,
#                         "thr_t": thr_t_out,
#                         "thr_mask": thr_mask,
#                     }
#                 else:
#                     thr_t_out = threshold(t_out, p_out, thr)
#                     out = {"beta": b_out, "t": t_out, "p": p_out, "thr_t": thr_t_out}
#             else:
#                 raise ValueError(
#                     "threshold_dict is not a dictionary. "
#                     "Make sure it is in the form of {'unc': .001} "
#                     "or {'fdr': .05}"
#                 )
#         else:
#             out = {"beta": b_out, "t": t_out, "p": p_out}

#         return out

#     def iplot(self, threshold=0, surface=False, anatomical=None, **kwargs):
#         """Create an interactive brain viewer for the current brain data instance.

#         Args:
#             threshold: (float/str) two-sided threshold to initialize the
#                         visualization, maybe be a percentile string; default 0
#             surface: (bool) whether to create a surface-based plot; default False
#             anatomical: nifti image or filename to overlay
#             kwargs: optional arguments to nilearn.view_img or
#                     nilearn.view_img_on_surf

#         Returns:
#             interactive brain viewer widget

#         """
#         if anatomical is not None:
#             if not isinstance(anatomical, nib.Nifti1Image):
#                 if isinstance(anatomical, str) or isinstance(anatomical, Path):
#                     anatomical = nib.load(anatomical)
#                 else:
#                     raise ValueError("anatomical is not a nibabel instance")
#         else:
#             # anatomical = nib.load(resolve_mni_path(MNI_Template)['brain'])
#             anatomical = get_mni_from_img_resolution(self, img_type="brain")
#         return plot_interactive_brain(
#             self, threshold=threshold, surface=surface, anatomical=anatomical, **kwargs
#         )

#     def plot(
#         self,
#         limit=5,
#         anatomical=None,
#         view="axial",
#         colorbar=False,
#         black_bg=True,
#         draw_cross=False,
#         threshold_upper=None,
#         threshold_lower=None,
#         figsize=(15, 2),
#         axes=None,
#         **kwargs,
#     ):
#         """Create a quick plot of self.data.  Will plot each image separately

#         Args:
#             limit: (int) max number of images to return
#             anatomical: (nifti, str) nifti image or file name to overlay
#             view: (str) 'axial' for limit number of axial slices;
#                         'glass' for ortho-view glass brain; 'mni' for
#                         multi-slice view mni brain; 'full' for both glass and
#                         mni views
#             threshold_upper: (str/float) threshold if view is 'glass',
#                              'mni', or 'full'
#             threshold_lower: (str/float)threshold if view is 'glass',
#                              'mni', or 'full'
#             save: (str/bool): optional string file name or path for saving; only applies if view is 'mni', 'glass', or 'full'.
#                             Filenames will appended with the orientation they belong to

#         """

#         if view == "axial":
#             if threshold_upper is not None or threshold_lower is not None:
#                 print("threshold is ignored for simple axial plots")
#             if anatomical is not None:
#                 if not isinstance(anatomical, nib.Nifti1Image):
#                     if isinstance(anatomical, str) or isinstance(anatomical, str):
#                         anatomical = nib.load(anatomical)
#                     else:
#                         raise ValueError("anatomical is not a nibabel instance")
#             else:
#                 # anatomical = nib.load(resolve_mni_path(MNI_Template)['plot'])
#                 anatomical = get_mni_from_img_resolution(self, img_type="plot")

#             if self.data.ndim == 1:
#                 if axes is None:
#                     _, axes = plt.subplots(nrows=1, figsize=figsize)
#                 plot_stat_map(
#                     self.to_nifti(),
#                     anatomical,
#                     cut_coords=range(-40, 60, 10),
#                     display_mode="z",
#                     black_bg=black_bg,
#                     colorbar=colorbar,
#                     draw_cross=draw_cross,
#                     axes=axes,
#                     **kwargs,
#                 )
#             else:
#                 if axes is not None:
#                     print("axes is ignored when plotting multiple images")
#                 n_subs = np.minimum(self.data.shape[0], limit)
#                 _, a = plt.subplots(
#                     nrows=n_subs, figsize=(figsize[0], len(self) * figsize[1])
#                 )
#                 for i in range(n_subs):
#                     plot_stat_map(
#                         self[i].to_nifti(),
#                         anatomical,
#                         cut_coords=range(-40, 60, 10),
#                         display_mode="z",
#                         black_bg=black_bg,
#                         colorbar=colorbar,
#                         draw_cross=draw_cross,
#                         axes=a[i],
#                         **kwargs,
#                     )
#             return
#         elif view in ["glass", "mni", "full"]:
#             if self.data.ndim == 1:
#                 return plot_brain(
#                     self,
#                     how=view,
#                     thr_upper=threshold_upper,
#                     thr_lower=threshold_lower,
#                     **kwargs,
#                 )
#             else:
#                 raise ValueError(
#                     "Plotting in 'glass', 'mni', or 'full' views only works with a 3D image"
#                 )
#         else:
#             raise ValueError("view must be one of: 'axial', 'glass', 'mni', 'full'.")
