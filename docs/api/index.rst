API reference
=============

Every symbol exported by ``structboost``, grouped by the task it belongs to.
This page is kept in sync with ``structboost.__all__`` by
``tests/test_public_api.py``.

For prose explaining when and why to use each of these, see the
:doc:`user guide <../guide/index>`.

The model
---------

The Boosting Autoencoder itself and its configuration. ``BAE`` carries the whole
fitting and inference surface: ``fit``, ``transform``, ``reconstruct``,
``stability_selection``, ``from_reference``, ``save`` and ``load`` are all
methods on it.

.. autosummary::
   :toctree: generated
   :recursive:

   structboost.BAE
   structboost.BAEConfig
   structboost.TrainingReport

Gene selection and reliability
------------------------------

How reproducible is the gene list? :doc:`../guide/tasks/gene-selection` explains
the two modes and what neither of them gives you.

Note that ``stability_selection`` below is the standalone ``allboost``-level
function, which does subsample mode only and has its own defaults. The method
:meth:`structboost.BAE.stability_selection` is the one you want for a fitted
model.

.. autosummary::
   :toctree: generated
   :recursive:

   structboost.stability_selection
   structboost.StabilitySelectionResult

Reconstruction quality
----------------------

A reconstruction MSE is uninterpretable on its own. ``linear_ceiling`` gives it a
reference. See :doc:`../guide/concepts/reading-quality`.

.. autosummary::
   :toctree: generated
   :recursive:

   structboost.linear_ceiling

Covariates and batch integration
--------------------------------

The encoding machinery behind ``condition_obs`` and ``nuisance_obs``. Most users
never call these directly, since ``fit`` does, but they are public so a design matrix
can be inspected or reused. See :doc:`../guide/tasks/batch-integration`.

.. autosummary::
   :toctree: generated
   :recursive:

   structboost.encode_obs_covariates
   structboost.transform_obs_covariates
   structboost.ObsCovariateEncoding

Transfer and encoder weight files
---------------------------------

An encoder weight matrix is the transferable product of a fit. These read and
write it as a standalone file so it can be carried to another dataset with
:meth:`structboost.BAE.from_reference`. Parquet is the recommended format:
spreadsheet round-trips silently rewrite gene symbols such as ``SEPT2`` and
``MARCH1`` as dates. See :doc:`../guide/tasks/transfer`.

.. autosummary::
   :toctree: generated
   :recursive:

   structboost.read_encoder_weights
   structboost.write_encoder_weights
   structboost.looks_like_ensembl

Interpretation
--------------

Turning a fitted encoder into something a biologist can read. See
:doc:`../guide/tasks/interpreting`.

.. autosummary::
   :toctree: generated
   :recursive:

   structboost.extract_gene_rankings
   structboost.write_annotations_to_h5ad
   structboost.DimensionAnnotation
   structboost.DimensionGeneRanking
   structboost.GeneRanking
   structboost.export_interactive_html

Plotting
--------

Require the ``[plot]`` extra.

.. autosummary::
   :toctree: generated
   :recursive:

   structboost.plot_training_diagnostics
   structboost.plot_boosting_coefficient_paths
   structboost.plot_top_boosting_coefficients

Boosting
--------

Componentwise L2 boosting on its own: sparse supervised learning with no
autoencoder involved, and the routine that fits the BAE encoder. See
:doc:`../guide/tasks/allboost`.

.. autosummary::
   :toctree: generated
   :recursive:

   structboost.allboost
   structboost.AllboostHistory
   structboost.compute_covariance_cache

Simulation
----------

Synthetic data with known marker genes, for scoring a method against ground
truth. See :doc:`../guide/tasks/simulating`.

.. autosummary::
   :toctree: generated
   :recursive:

   structboost.sim_scrnaseq_data
   structboost.sim_scrnaseq_anndata
   structboost.SimulationResult
