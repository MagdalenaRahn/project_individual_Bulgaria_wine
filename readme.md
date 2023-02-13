This respository contains functions and code for the individual project at Codeup, Data Science programme, Noether cohort, 13 February 2023.


Hypotheses:
1/ There is no relationship between the weather and the coupon being accepted.
H_a : There is a relationship between the weather and the coupon being accepted.

2/ People heading in the opposite direction are equally likely to accept the coupon as people heading in the same direction as the coupon.
H_a : People heading in the opposite direction are less likely to accept the coupon than people heading in the same direction as the coupon.

3. People who make $74999 or less are equally likely to accept the coupon as people who make $75000 or more.
H_a : People who make $74999 or less are more likely to accept the coupon as people who make $75000 or more.

4. There is no relationship between having children and accepting the coupon.
H_a : There is a relationship between having children and accepting the coupon.



Data Dictionary :

destination: No Urgent Place, Home, Work 

passanger: Alone, Friend(s), Kid(s), Partner (who are the passengers in the car)  

weather: Sunny, Rainy, Snowy  

temperature: 55, 80, 30  

time: 2PM, 10AM, 6PM, 7AM, 10PM  

coupon: Restaurant(<$20), Coffee House, Carry out & Take away, Bar, Restaurant($20-$50)  

expiration: 1d, 2h (the coupon expires in 1 day or in 2 hours)  

gender: Female, Male  

age: 21, 46, 26, 31, 41, 50plus, 36, below21 

marital_status: Unmarried partner, Single, Married partner, Divorced, Widowed  

has_children: 1, 0  

education: Some college - no degree, Bachelors degree, Associates degree, High School Graduate, Graduate degree (Masters or Doctorate), Some High School  

occupation: Unemployed, Architecture & Engineering, Student, Education&Training&Library, Healthcare Support, Healthcare Practitioners & Technical, Sales & Related, Management, Arts Design Entertainment Sports & Media, Computer & Mathematical,
Life Physical Social Science, Personal Care & Service, Community & Social Services, Office & Administrative Support, Construction & Extraction, Legal, Retired, Installation Maintenance & Repair, Transportation & Material Moving, Business & Financial, Protective Service, Food Preparation & Serving Related, Production Occupations, Building & Grounds Cleaning & Maintenance, Farming Fishing & Forestry

income: $37500 - $49999, $62500 - $74999, $12500 - $24999, $75000 - $87499, $50000 - $62499, $25000 - $37499, $100000 or More, $87500 - $99999, Less than $12500  

bar: never, less1, 1~3, gt8, nan4~8 (feature meaning: how many times do you go to a bar every month?)  

coffee_hse: never, less1, 4~8, 1~3, gt8, nan (feature meaning: how many times do you go to a coffeehouse every month?)  

carry_away: n4~8, 1~3, gt8, less1, never (feature meaning: how many times do you get take-away food every month?)  

resto_less_than_20: 4~8, 1~3, less1, gt8, never (feature meaning: how many times do you go to a restaurant with an average 
expense per person of less than $20 every month?)  

resto_20_50: 1~3, less1, never, gt8, 4~8, nan (feature meaning: how many times do you go to a restaurant with average expense per person of $20 - $50 every month?)

to_coupon_GEQ15m: 0, 1 (feature meaning: driving distance to the restaurant/bar for using the coupon is greater than 15 minutes)

to_coupon_GEQ25m: 0, 1 (feature meaning: driving distance to the restaurant/bar for using the coupon is greater than 25 minutes)

direction_same: 0, 1 (feature meaning: whether the restaurant/bar is in the same direction as your current destination)

direction_opp: 1, 0 (feature meaning: whether the restaurant/bar is in the same direction as your current destination)

Y: 1, 0 (whether the coupon is accepted)
