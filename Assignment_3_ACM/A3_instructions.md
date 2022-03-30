In assignment 3 you have to analyze real world data using Bayesian models of cognition.

The data comes from the social conformity experiment (https://pubmed.ncbi.nlm.nih.gov/30700729/), where cogsci students (in dataset 1) and schizophrenia patients + controls (dataset 2) combine their own intuition of trustworthiness of given faces to social information.

Your task is to

implement 2 models (at least): simple Bayes vs weighted Bayes (bonus points if you also include a GLM model).
fit them to one dataset (don't forget to explore the data first!)
check model quality
do model comparison
report (v minimal description of research question, v minimal description of data, description of models, model quality checks, report of results)
[optional]: parameter/model recovery
Guide to the social conformity data

Both datasets (students and schizophrenia) include the following columns:

- FaceID: an identifier of the specific face rated
- ID: an identifier of the participant
- Trial_Round1: in which trial the face was presented (during the first exposure)
- Trial_Round2: in which trial the face was presented (during the second exposure)
- FirstRating: the trustworthiness rating (1-8) given by the participant BEFORE seeing other ratings   
- OtherRating: the trustworthiness rating (1-8) given by others
- SecondRating: the trustworthiness rating (1-8) given after seeing the others (at second exposure) The students dataset also includes:

- Change: the difference between the second and the first rating - Class: participants belong to two different cohorts (1 and 2) tested at different times - Feedback: the difference between other rating and own first rating

The schizophrenia dataset also includes: - Group: 0 is comparison group, 1 is schizophrenia group - RT_Round1: time taken to produce the first rating of trustworthiness
- RT_Round2: time taken to produce the second rating of trustworthiness

The schizophrenia data was collected within the study described in Simonsen, A., Fusaroli, R., Skewes, J. C., Roepstorff, A., Mors, O., Bliksted, V., & Campbell-Meiklejohn, D. (2019). Socially learned attitude change is not reduced in medicated patients with schizophrenia. Scientific reports, 9(1), 1-11.

The students data is currently unpublished.

Student data: https://www.dropbox.com/s/r9917ta89qhwsl2/sc_students.csv?dl=0

Patient data: https://www.dropbox.com/s/td2oeos6kfrx7td/sc_schizophrenia.csv?dl=0