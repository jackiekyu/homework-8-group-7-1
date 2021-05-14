# Analysis Part III

Imagine enumerating all the subject-weeks (4 weeks per subject), and labeling each as either “has symptoms” or “does not have symptoms,” so that each week is labeled with a binary outcome. Can either of the methods you implemented be used to find confidence bounds for the average treatment effect, measured as the reduction in the average number of symptomatic weeks per subject? If so, explain how; if not, explain why.

#### Can either of the methods you implemented be used to find confidence bounds for the average treatment effect?

The methods implemented cannot be used to find confidence bounds for the average treatment effect. In this case, the assumption of independence does not hold. Whether or not a subject has symptoms the next week can be dependent on whether or not they had symptoms in the previous week.