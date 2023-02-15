This respository contains functions and code for the individual project at Codeup, Data Science programme, Noether cohort, 13 February 2023.  

**Author :** Magdalena Rahn  

**Project Title :** Will Drive For Coupons  

**Project Description :**
1/ This project aims to predict the driver of a vehicle accpeting a coupon when it is offered on his / her smartphone while at the wheel. The data represents characteristics of the vehicle driver and situation (weather, time, location, distance from coupon location, gender, occupation, etc) and this person's response to being offered a limited-time coupon on a mobile application. Some of these features were dropped early on, to allow for focussed analysis. The goal was to predict which categories of persons would accept, or not, which types of coupons.    

2/ The target variable is 'Y', which is a boolean of no (0) / yes (1) as to whether the driver accepted the coupon.  

3/ Data comes from UCI, via 'The Journal of Machine Learning Research' (JMLR). (This journal, established in 2000, is an international forum for the electronic and paper publication of scholarly articles in all areas of machine learning. Original research paper : https://jmlr.org/papers/v18/16-003.html.) Data was collected in 2015 using Amazon Mechanical Turk. 
UCI Machine Learning data repository : https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation.  
  
  

**QUESTIONS TO EXPLORE :**  
- Question 1 : Is there a relationship between the weather and the coupon being accepted ?   

- Question 2 : Are people heading in the opposite direction of the coupon offer more likely to accept the coupon than people heading in the same direction as the coupon ?  

- Question 3 : Is there a relationship between income group and accepting the coupon ?   

- Question 4 : How does frequency of visiting a bar relate to accepting the coupon ?   


**HYPOTHESES :**  
1/  H_0: There is no relationship between the weather and the coupon being accepted.   
H_a : There is a relationship between the weather and the coupon being accepted.   

2/ H_0 : People heading in the opposite direction are equally likely to accept the coupon as people heading in the same direction as the coupon.  
H_a : People heading in the opposite direction are more likely to accept the coupon than people heading in the same direction as the coupon.  

3/ H_0 : Income group has no relationship to accepting the coupon.  
𝐻𝑎 : Income group has a relationship to accepting the coupon.  

4/ H_0: There is no relationship between bar-visit frequency and accepting the coupon.   
H_a : There is a relationship between bar-visit frequency and accepting the coupon.  



**The Process / Project Plan :**   
1. Obtain, explore and analyse wine chemical composition and wine quality data from the UCI Machine Learning Database, with 'Y' as the target variable. Do this using simple Python coding.  

2. Analyse features in the tidied data based on (1) relationship between the weather and the coupon being accepted, (2) weather people heading in the opposite direction of the coupon offer more likely to accept the coupon than people heading in the same direction as the coupon, (3) whether there a relationship between income group and accepting the coupon and (4) how frequency of visiting a bar relate to accepting the coupon.  
These were analysed against the target variable of accepting the coupon or not, 'Y'.  

3. Model the data using comparative visualisations in Python (Seaborn and MatPlotLib).  

4. Apply statistical modelling in Python to select data to determine mathematical probability as compared with visual indications.  

5. Run classification and linear regression models on the data based on earlier findings. Analyse these results.  

6. Provide suggestions and indicate next steps that could be performed.  




**Data Dictionary :**  

- weather: Sunny, Rainy, Snowy  

- temperature: 55, 80, 30  

- time: 2PM, 10AM, 6PM, 7AM, 10PM  

- coupon: Restaurant(<$20), Coffee House, Carry out & Take away, Bar, Restaurant($20-$50)  

- expiration: 1d, 2h (the coupon expires in 1 day or in 2 hours)  

- gender: Female, Male  

- age: 21, 46, 26, 31, 41, 50plus, 36, below21 

- marital_status: Unmarried partner, Single, Married partner, Divorced, Widowed  

- has_children: 1, 0  

- education: Some college - no degree, Bachelors degree, Associates degree, High School Graduate, Graduate degree (Masters or Doctorate), Some High School  

- occupation: Unemployed, Architecture & Engineering, Student, Education&Training&Library, Healthcare Support, Healthcare Practitioners & Technical, Sales & Related, Management, Arts Design Entertainment Sports & Media, Computer & Mathematical,
Life Physical Social Science, Personal Care & Service, Community & Social Services, Office & Administrative Support, Construction & Extraction, Legal, Retired, Installation Maintenance & Repair, Transportation & Material Moving, Business & Financial, Protective Service, Food Preparation & Serving Related, Production Occupations, Building & Grounds Cleaning & Maintenance, Farming Fishing & Forestry

- income: $37500 - $49999, $62500 - $74999, $12500 - $24999, $75000 - $87499, $50000 - $62499, $25000 - $37499, $100000 or More, $87500 - $99999, Less than $12500  

- bar: never, less1, 1~3, gt8, nan4~8 (feature meaning: how many times do you go to a bar every month?)  

- coffee_hse: never, less1, 4~8, 1~3, gt8, nan (feature meaning: how many times do you go to a coffeehouse every month?)  

- carry_away: n4~8, 1~3, gt8, less1, never (feature meaning: how many times do you get take-away food every month?)  

- resto_less_than_20: 4~8, 1~3, less1, gt8, never (feature meaning: how many times do you go to a restaurant with an average 
expense per person of less than $20 every month?)  

- resto_20_50: 1~3, less1, never, gt8, 4~8, nan (feature meaning: how many times do you go to a restaurant with average expense per person of $20 - $50 every month?)

- to_coupon_GEQ15m: 0, 1 (feature meaning: driving distance to the restaurant/bar for using the coupon is greater than 15 minutes)

- to_coupon_GEQ25m: 0, 1 (feature meaning: driving distance to the restaurant/bar for using the coupon is greater than 25 minutes)

- direction_same: 0, 1 (feature meaning: whether the restaurant/bar is in the same direction as your current destination)

- direction_opp: 1, 0 (feature meaning: whether the restaurant/bar is in the same direction as your current destination)

- Y: 1, 0 (whether the coupon is accepted)



**For Further Exploration :** 
1/ Combine features such as weather & income group, or how frequently the driver visits a coffee house & education level, to explore how they relate to the target variable.  



**Steps To Reproduce :**    
1. Assure the presence of a Python environment on your computer.  

2. Import :  
- Python libraries pandas, numpy, matplotlib, seaborn and scipy,   
- The red and white Wine Quality databases from https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation and save the file locally, and  
- Pre-existing or self-created data 'acquire' and 'fonction' modules.  

3. Tidy the data.  Make numerous dummies.  

4. Explore using graphs, statistical evaluation, feature engineering, and modelling.  

5. Evaluate, analyse and form conclusions and recommendations and indicate next steps.  

