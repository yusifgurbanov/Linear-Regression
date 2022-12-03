#libraries 
import pandas  # for data read, data analyze
import numpy  # for mathematical operations such as linear algebra, regressions
import matplotlib.pyplot as mp # for data visualization
from mpl_toolkits.mplot3d import Axes3D # for showing data in 3D
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model # for Linear Regression
from sklearn.preprocessing import PolynomialFeatures # for linear regression by polynomial equation 

file = pandas.read_csv('turboaz.csv')
data = file[['Buraxilish ili', 'Yurush', 'Qiymet']]
#print(data) # print 1329 rows x 3 columns ('Buraxilish ili', 'Yurush', 'Qiymet')

# X1  “Yurush”  = column1
# X2  “Buraxilish ili”  = column2
# Y  “Qiymet”  = y

#-------------------------Loading data--------------------------------
column1 = file['Yurush'].map(lambda i: i.rstrip('km').replace(' ', '')).map(int) # remove string and convert to number
#print(column1) #print 1328 lines data of Yurush column

column2 = file['Buraxilish ili']
#print(column2) #print 1328 lines data of Buraxilish ili column

y = file['Qiymet'].map(lambda i: float(i.rstrip('$'))*1.7 if '$' in i else float(i.rstrip('AZN'))) # remove AZN and $. If $, convert into AZN
#print(y) #print 1328 lines data of Qiymet column

#save values of column1, column2, and y since they will be changed for operations
save_column1 = column1
save_column2 = column2
save_y = y


# ---------------------------Visualization------------------------------------------------
# Qiymet (Y) vs Yurush (column1)
mp.scatter(column1, y, color='blue') 
mp.xlabel('Mileage')
mp.ylabel('Price')
mp.title('Mileage vs Price')
mp.show()

# Qiymet (Y) vs Buraxilish ili (column2) 
mp.scatter(column2, y, color='blue') 
mp.xlabel('Year')
mp.ylabel('Price')
mp.title('Year vs Price')
mp.show()

# 3D plot of all three values (Y, column1, column2) 
graph = mp.figure().add_subplot(projection='3d')
graph.scatter(column1, column2, y, color='blue')
graph.set_xlabel('Mileage')
graph.set_ylabel('Year')
graph.set_zlabel('Price')
mp.title('Mileage vs Year vs Price')
mp.show()


#-----------------Implementation of Linear Regression from scratch----------------
#cost function
def cost_function(x, y, z): 
    formula = numpy.sum((x.dot(z) - y) ** 2) / (2 * len(y)) #formula =1/2n * sum(((w^T)*x) - y)^2
    return formula

#normalization (mean = average)
column1 = (column1 - column1.mean()) / column1.std() 
column2 = (column2 - column2.mean()) / column2.std()
y = (y - y.mean()) / y.std()

#gradient descent algorithm
alpha = 0.001
iterations = 10000
def gradient(x, y, alpha, iterations):
    cost = [] 
    
    mean = x.mean(axis=0)
    standart_deviation = x.std(axis=0) #standart deviation
    x = (x - mean) / standart_deviation

    x = numpy.append(x, numpy.ones((len(file), 1)), axis=1)
    random = numpy.random.random(x.shape[1]) # assign random values

    for i in range(iterations):
        h = numpy.matmul(x, numpy.array(random))
        random = random - alpha * (1/len(y)) * (numpy.matmul(x.transpose(), h-y)) #Gradient descent formula
        cost.append(cost_function(x, y, random)) #collect cost at each iteration
    
    return random, cost, mean, standart_deviation

file['Yurush'] = save_column1
file['Qiymet'] = save_y

column1_column2 = numpy.array(file[['Yurush', 'Buraxilish ili']])
y1 = numpy.array(file['Qiymet'])

random, cost, mean, standart_deviation = gradient(column1_column2, y1 , alpha, iterations)

mp.plot(cost)
mp.xlabel('Iterations')
mp.ylabel('Cost')
mp.title("Cost per Iteration")
mp.show() # graph depends on the number of iterantions

# Plot points of Y (Qiymet) vs column1 (Buraxilish ili) 
mp.scatter(file["Buraxilish ili"], file["Qiymet"], color='blue')
x = [1990, 2022]
y = [random[2] + (1990 - mean[1]) / standart_deviation[1] * random[1],  random[2] + (2022 - mean[1]) / standart_deviation[1] * random[1]]
mp.plot(x, y, color='red')
mp.title('Year vs Price')
mp.show()

# Plot points of Y (Qiymet) vs column2 (Yurush) 
mp.scatter(x=file['Yurush'], y=file['Qiymet'], color='blue')
x = [0, 1000000]
y = [random[2], random[2] + (1000000 - mean[0]) / standart_deviation[0] * random[0]]
mp.plot(x, y, color="red")
mp.title('Mileage vs Price')
mp.show()

# Plot 3D graph of points of Y (Qiymet), column1, column2  and predicted Y (Qiymet) 
arr = file[["Yurush", "Buraxilish ili"]]
arr = (arr - mean) / standart_deviation # true values
arr1 = numpy.ones((arr.shape[0], 1))
arr = numpy.append(arr, arr1, axis=1) 

h = arr.dot(random) # predicted value

graph1 = mp.figure().add_subplot(projection='3d')
graph1.scatter(file["Yurush"], file["Buraxilish ili"], file["Qiymet"], color="blue")
graph1.scatter(file["Yurush"], file["Buraxilish ili"], h, color="red")
graph1.set_xlabel('Mileage')
graph1.set_ylabel('Year')
graph1.set_zlabel('Price')
mp.title('Mileage vs Price vs Mileage')
mp.show()


#------------------------Testing----------------------------
test_price = numpy.array([11500, 8800])
test_mileage_year = numpy.array([[240000, 2000], [415558, 1996]])

mean2 = numpy.array([mean[0], mean[1]])
standart_deviation2 = numpy.array([standart_deviation[0], standart_deviation[1]])

test_mileage_year = (test_mileage_year - mean2) /standart_deviation2 #normalization
arr2 = numpy.ones((len(test_mileage_year), 1))
test_mileage_year = numpy.append(test_mileage_year, arr2, axis=1)

predicted_price = test_mileage_year.dot(random)
gap = predicted_price - test_price 

print('First car')
print("Actual Price:", test_price[0], "; Predicted Price:",predicted_price[0], "; Difference:", gap[0])
print("Second car")
print("Actual Price:", test_price[1], "; Predicted Price:",predicted_price[1], "; Difference:", gap[1])


#------------------Linear Regression using library----------------------
model = linear_model.LinearRegression().fit(file[['Yurush', 'Buraxilish ili']], file.Qiymet)

test_price1 = numpy.array([11500, 8800])
test_mileage_year1 = [[240000, 2000], [415558, 1996]]

predicted_price1 = model.predict(test_mileage_year1)
gap1 = predicted_price1 - test_price1

print('First car')
print("Actual Price:", test_price1[0], "; Predicted Price:", predicted_price1[0], "; Difference:", gap1[0])
print("Second car")
print("Actual Price:", test_price1[1], "; Predicted Price:", predicted_price1[1], "; Difference:", gap1[1])


#-----------------------Linear regression by Normal equation -----------------------
arr3 = numpy.ones((len(file[["Yurush", "Buraxilish ili"]].values), 1))

norm_x= numpy.append(file[["Yurush", "Buraxilish ili"]].values, arr3, axis=1)
norm_y = file["Qiymet"].values 

# formula = (X^T . X)^-1 . X^T . y
answer = numpy.linalg.inv(norm_x.T.dot(norm_x)).dot(norm_x.T).dot(norm_y)
print("Linear Regression by Normal Equation:",answer)


#--------------------Polynomial function--------------------
po = PolynomialFeatures(2) # with degree 2
li = linear_model.LinearRegression() # linear regression model

#fit the polynomial model into linear regression model
li.fit(po.fit_transform(file[["Yurush", "Buraxilish ili"]].values), file['Qiymet']) 

