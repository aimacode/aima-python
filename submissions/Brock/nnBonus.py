# (75) C level achieved - 5844 samples and 118 input variables, with 12 output variables
#     month with 55% correct and 45% errors, two layers, and a target set

# inputs and targets are parsed from the nnInput.csv and nnTarget.csv files

# Bonuses achieved:

# (5) 1000+ samples - I have 5844
# (5) 64+ input variables - I have 118
# (5) another 2-layer submission, that reduces errors by at least 20%
#    - month2, decreases errors by 27% from 45% to 33%, I trained the weights with 2 extra epochs
# (5) a 3-layer submission, that reduces errors by at least 20%
#    - month3, decreases errors by 37% from 33% to 21%, increased optimizer by 10x, and utilized
#      categorical_crossentropy loss function
# (5) a 4-layer submission, that reduces errors by at least 20%
#    - month4, decreases errors by 62% from 21% to 8%, increased optimizer by another 10x, and utilized the
#      hard_sigmoid function