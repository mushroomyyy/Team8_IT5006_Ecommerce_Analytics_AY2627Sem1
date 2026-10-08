# Cross-validation and hyperparameter tuning

(Draft for the report, June development and selection, followed by frozen July–August evaluation.)

**Initial model configurations.** Each model was first evaluated using a fixed starting configuration, providing a reference against which to assess hyperparameter tuning:

- **Linear models:** Logistic Regression used C = 1; regression used Linear Regression and Ridge with alpha = 1.
- **Decision Tree:** classification used a maximum depth of 10; regression imposed no depth limit.
- **Random Forest:** both tasks used 100 trees with a maximum depth of 10.
- **XGBoost:** classification used 100 trees, a maximum depth of 6 and a learning rate of 0.1.
- **LightGBM:** both tasks used 100 trees, a learning rate of 0.1 and 31 leaves.

For classification, Logistic Regression, Decision Tree and Random Forest used balanced class weights. XGBoost and LightGBM recalculated the late-class weight from the training labels at each fit. Other settings followed the implementation.

**Chronological cross-validation.** Both analyses used a 365-day historical window covering 18 April 2017–17 April 2018 for the June run. The approval cutoff was 45 days before June inference. The most recent 30 days, 19 March–17 April 2018, formed the development holdout. A further 45-day gap separated this holdout from the earlier data used to fit and tune candidate models. Training orders were retained only when their outcomes were available at the start of the holdout. Default and tuned candidates used identical training and holdout orders within each task, allowing their performance to be compared on a consistent basis.

Within the earlier training period, five expanding chronological folds were constructed, as illustrated in the validation timeline figure. Each fold trained on earlier orders and validated on a subsequent 30-calendar-day block, separated by a 45-day gap. The first fold had approximately 95 calendar days of initial training history; later folds incorporated progressively more data. Orders approved on the same day remained together, and training outcomes were checked for availability at each validation start. Imputation, encoding and scaling, where required, were fitted only on the training portion of each fold. This arrangement reflects the information available when predicting later orders.

**Hyperparameter tuning.** Random search was implemented using RandomizedSearchCV with seed 42. Model-specific search spaces covered regularisation strength, tree depth and leaf size, ensemble size, learning rate and sampling settings, as applicable. Regression LightGBM also compared squared-error and absolute-error objectives. Classification selected hyperparameters by maximising mean average precision (AP) across the five folds, while regression minimised mean absolute error (MAE), measured in days.

Classification evaluated 12 configurations for Logistic Regression, 15 for Decision Tree, eight for Random Forest, and 12 each for XGBoost and LightGBM. Regression evaluated 12 configurations for Ridge, 15 for Decision Tree, six for Random Forest and 10 for LightGBM. Linear Regression was evaluated without tuning. The notebooks specify the complete candidate values.

**Candidate selection and frozen evaluation.** Initial and tuned candidates were evaluated on the same buffered development holdout. The best-on-holdout candidate was recorded using AP for classification (with F1 and ROC-AUC resolving ties) and MAE for regression. This comparison provided a development reference; it did not determine the final deployment candidate. The classification threshold remained 0.5.

All candidates were then refitted on the full eligible historical window and used to score June orders. Daily predictions were pooled across June, and the candidate with the highest June AP or lowest June MAE was selected for classification or regression, respectively. June therefore served as an additional model-selection cohort, and its reported performance should not be described as an untouched test estimate. Feature-importance analysis was conducted for the June-selected candidate.

The selected fitted model, including its preprocessing and parameter values, was subsequently applied to July and August without retraining, retuning or reselection. Monthly metrics describe the stability of this fixed model under later delivery conditions. Changes in performance may reflect changes in the order population and delivery process; monthly differences alone do not establish the cause of deterioration.

**Timing of retrospective evaluation.** Outcomes were assessed at approval plus 45 days. Complete assessment of the last June approvals reaches 14 August and is available from 15 August under the implementation's strict-before-run rule. Accordingly, July and early-August predictions are retrospective frozen-model checks, rather than a fully prospective simulation of deploying the June-selected model on 1 July. July and August outcomes were not used to select or update that model.

**Final CV settings and selected candidates.** CV-selected settings describe the tuned candidate within each model family; June selection compares both starting and tuned candidates. The classification searches returned:

- **Logistic Regression:** C = 3162.28.
- **Decision Tree:** min_samples_leaf = 20, max_depth = 12, criterion = entropy.
- **Random Forest:** n_estimators = 200, min_samples_leaf = 1, max_features = sqrt, max_depth = 12.
- **XGBoost:** subsample = 0.7, n_estimators = 600, min_child_weight = 10, max_depth = 2, learning_rate = 0.01, colsample_bytree = 1.
- **LightGBM:** subsample_freq = 1, subsample = 1, reg_lambda = 0, num_leaves = 7, n_estimators = 400, min_child_samples = 20, learning_rate = 0.05, colsample_bytree = 0.6.

The June-selected classifier was the starting LightGBM configuration (100 trees, learning rate 0.1 and 31 leaves), rather than its CV-tuned variant. Its pooled June AP was 15.09%; the frozen July and August AP values were 12.77% and 11.34%, respectively. Classification metrics used 6,155, 6,125 and 6,606 evaluable orders from 6,164, 6,129 and 6,606 scored orders, respectively. The historical holdout separately favoured tuned Logistic Regression (AP 24.20%).

**Regression CV settings.** The regression searches returned:

- **Ridge:** alpha = 1000.
- **Decision Tree:** minimum leaf size = 100; maximum depth = 6.
- **Random Forest:** 200 trees; minimum leaf size = 60; maximum features = 0.5; maximum depth = 16.
- **LightGBM:** absolute-error objective (`regression_l1`); 600 trees; 63 leaves; minimum child samples = 50; learning rate = 0.02; row sampling = 0.7 with sampling frequency = 1; feature sampling = 0.6.

Linear Regression retained its untuned starting configuration. Tuned LightGBM was selected both on the development holdout (MAE 5.3945 days) and on June orders (MAE 4.8455 days). Its frozen July and August MAE values were 4.6117 and 4.5910 days. These monthly metrics used 6,164, 6,159 and 6,612 evaluable orders from 6,164, 6,176 and 6,620 scored orders, respectively.

Suggested figure caption: **Chronological development and June model selection.**
