import pandas as pd
import numpy as np

def genirate_data():

    assignment_cols = []
    for i in range(1,11):
        assignment_cols.append(f'assignment_{i}')

    class_ids = []
    stu_ids = []
    for i in range(1,7):
        for j in range(1,31):
            class_ids.append(i)
            stu_ids.append(j)

    new_data = pd.DataFrame(columns=['class_id','stu_id','participate','attendance']+assignment_cols)
    new_data['class_id'] = class_ids
    new_data['stu_id'] = stu_ids

    for assignment in assignment_cols:
        new_data[assignment] = np.random.randint(0,101,180)
    new_data['participate'] = np.random.randint(25,76,180)
    new_data['attendance'] = np.random.randint(0,2,180)

    new_data.to_csv('new_data.csv',index=False)

    participate = pd.DataFrame(columns=['class_id','required_participate'])
    participate['class_id'] = list(set(class_ids))
    participate['required_participate'] = np.random.randint(25,76,6)

    participate.to_csv('participate.csv',index=False)

    week = []
    for i in range(1,8):
        week.append(f'day_{i}')

    performance = pd.DataFrame(columns=['class_id','last_average_performance','last_delta']+week)
    performance.class_id = participate.class_id
    performance.last_delta = np.random.uniform(-25,26,6)
    for day in week:
        performance[day] = np.random.uniform(10,91,6)
    performance.last_average_performance = performance[week].T.mean().values

    performance.to_csv('perfomance.csv',index=False)

def calculate():

    new_data = pd.read_csv('new_data.csv')
    new_data_ = new_data.iloc[:,:-10].copy()
    new_data_['avg_stu_assignment_performance'] = new_data.iloc[:,-10:].T.mean().values

    participate = pd.read_csv('participate.csv')

    new_data_ = pd.merge(new_data_,participate,on='class_id',how='left')

    new_data_['participated'] = new_data_.participate / new_data_.required_participate
    new_data_.participated[new_data_.participated > 1] = 1

    new_data_['stu_performance'] = new_data_.avg_stu_assignment_performance*0.8 + new_data_.participated*15 + new_data_.attendance*5
    class_performance = new_data_[['class_id','stu_performance']].groupby('class_id').mean().rename(columns={'stu_performance':'class_performance'})

    performance = pd.read_csv('perfomance.csv')

    performance.last_delta = class_performance.class_performance.values - performance.last_average_performance.values
    for i in range(1,7):
        performance[f'day_{i}'] = performance[f'day_{i+1}']
    performance.day_7 = performance.last_average_performance
    performance.last_average_performance = class_performance.class_performance.values

    performance.to_csv('last_update.csv')
        