#!/usr/bin/env python
# coding: utf-8

# # Part I. Functions

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from scipy import stats
import time


# In[2]:


SamSig_AC = pd.read_csv('Data/SampleSignal_AC.csv')
SamSig_IE = pd.read_csv('Data/SampleSignal_IE.csv')
Sec_Name = pd.read_csv('Data/sector_name.csv')
Sec_Trd = pd.read_csv('Data/TrdData_CISector.csv')


# In[3]:


def pct_ranking(date_str, factor_df, num_grps_int = 5, delay_int = 1):
    
    '''
    Works to divide the cross-sectional data into groups of differnet factor scores;
    Outputs a dataframe;
    date_str is the portfolio re-adjustment date per cycle of type str;
    factor_df should be a dataframe of n industries of size n with trading dates behaving like index;
    num_grps_int are the desired number of ranking groups of type int;
    delay is the number of days the factor index can be retrived
    '''

    pool = factor_df.loc[factor_df['TrdDt'] == factor_df['TrdDt'][factor_df['TrdDt'].index[factor_df['TrdDt'] == 
                                                                                           date_str][0]-delay_int]] 
    # assuming knowing the factor 1 trading day later
    
    cols = [i for i in pool.columns if i not in ['TrdDt']]
    pool = pool[cols]
    grp = pd.qcut(pool.iloc[0,:],num_grps_int, 
        labels = [f'Group{i}' for i in range(num_grps_int,0,-1)]
    )
    
    assignment = pd.DataFrame({
        'Score':pool.iloc[0,:].to_list(),
        'Group':grp
    })
    
    
    return assignment


# In[4]:


def TrdDates(trd_df):
    '''
    Returns a list of trading dates given any trading dataframe
    trd_df is any trading data of type dataframe
    '''
    
    TrdDt = trd_df['TrdDt'].unique()
    TrdDt = [dt.datetime.strptime(i, '%Y-%m-%d').date() for i in TrdDt]
    
    return TrdDt


# In[5]:


def convert_dt(date_freq_str,types = 'date'):
    '''
    Basic converting function utilizing datetime to make the final back-testing function more efficient
    '''
    if types == 'date':
        date = dt.datetime.strptime(date_freq_str, '%Y-%m-%d').date()
        return date
    elif types == 'delta':
        time_delt = dt.timedelta(date_freq_str)
        return time_delt
    else:
        print('Error')
    


# In[6]:


def rebalance_dates(TrdDt, start_dt, end_dt, freq_dt): # Natural Day Based Algo
    '''
    Works to find the schedule of rebalancing dates given back-testing range and frequency;
    outputs a list;
    TrdDt is the trading dates;
    start is the starting day of the back-testing of str format;
    end is the ending day of the back-testing of str format;
    freq is the frequency of the back-testing of str format in the unit of day;
    '''

    dates = [start_dt]
    list_1 = [0]+[1]*364
    for i in range(1,int((end_dt-start_dt)/freq_dt)+1): 
        date = dates[i-1]+freq_dt
        if date > end_dt:
            break
        for j in list_1: # if the readjustment date isn't trading day, shift to the next day
            date += dt.timedelta(list_1[j]) # shifting range is limited to 365d
            if date in TrdDt:
                if date > end_dt:
                    break
                dates.append(date)
                break
    
    return dates


# In[7]:


def rebalance_dates_v2(TrdDt, start_dt, end_dt, freq_str, assign = False): # Trading Day Based Algo
    '''
    Works to find the schedule of rebalancing dates given back-testing range and frequency;
    outputs a list;
    TrdDt is the trading dates;
    start is the starting day of the back-testing of str format;
    end is the ending day of the back-testing of str format;
    freq is the frequency of the back-testing of str format in the unit of day;
    '''
    if not assign:
        dates = []
        rebalance_list = np.arange(TrdDt.index(start_dt),TrdDt.index(end_dt), int(freq_str))
        for i in rebalance_list:
            dates.append(TrdDt[i])
    
    else: 
        dates = assign

    
    return dates


# In[8]:


def buy_sell_single(factor_df, TrdDt, start_dt, end_dt, freq_str, assign = False, num_grps_int=5, delay_int=1):
    '''
    The version without rolling windows. Figured would have too many for-loops if included.
    Will include the windows afterwards.
    '''

    dates_df = rebalance_dates_v2(TrdDt, start_dt, end_dt, freq_str, assign)

    schedule = pd.DataFrame()
    for i in dates_df:
        test_df = pct_ranking(str(i), factor_df, num_grps_int, delay_int)
        schedule = pd.concat([schedule, test_df.groupby('Group').apply(lambda x: x.index.tolist())], axis=1)
    
    column_names = [date.strftime('%Y-%m-%d') for date in dates_df]
    schedule.columns = column_names
    
    return schedule


# In[9]:


def sec_id(sec_name_str, sec_name_df):
    '''
    Given sector name outputs its corresponding sector id.
    For now, assume sector ID column is always named 'SecCd'
    '''
    sec_id = sec_name_df.loc[sec_name_df['SecName'] == sec_name_str, 'SecCd'].iloc[0]

    return sec_id


# In[10]:


def sec_name(sec_id_str, sec_name_df):
    '''
    Given sector ID outputs its corresponding sector name.
    For now, assume sector name column is always named 'SecName'
    '''
    sec_name = sec_name_df.loc[sec_name_df['SecCd'] == sec_id_str, 'SecName'].iloc[0]

    return sec_name


# In[11]:


def days_diff(dates_list,start_dt,end_dt):
   
    left = dates_list.index(start_dt)
    right = dates_list.index(end_dt)
    
    return right - left


# In[12]:


def date_diff(dates_list, start_dt, end_dt):

    left = dates_list.index(start_dt)
    right = dates_list.index(end_dt)
    
    daytime = []
    for i in dates_list[left+1:right]:
        daytime.append(i.strftime('%Y-%m-%d'))
    return daytime


# In[13]:


def max_drawdown(array):
    drawdowns = []
    max_so_far = array.iloc[1]
    for i in range(len(array)-1):
        if array.iloc[i+1] > max_so_far:
            drawdown = 0
            drawdowns.append(drawdown)
            max_so_far = array.iloc[i+1]
        else:
            drawdown = max_so_far - array.iloc[i+1]
            drawdowns.append(drawdown)
    return max(drawdowns)


# In[14]:


def roll_rebalance_dates(TrdDt, start_dt, end_dt, freq_str,assign = False, roll_window_int = 0):
    '''
    Adds another layer to the original rebalance_dates function, 
    allowing for sensitivity tests of the factor as well as the strategies' timeliness;
    roll_window_int is the integer of the desired rolling window of the rebalancing dates
    '''
    
    forward_dict = {} 
    backward_dict = {}

    if not assign:
        for i in range(roll_window_int + 1):
            if i == 0:
                Origin = rebalance_dates_v2(TrdDt, start_dt, end_dt, freq_str)
                continue
            forward_dict[f'{i}-Day After'] = rebalance_dates_v2(TrdDt, TrdDt[TrdDt.index(start_dt)+i], 
                                                                  TrdDt[TrdDt.index(end_dt)+i], freq_str)
        for i in range(roll_window_int,0,-1):
            backward_dict[f'{i}-Day Priv'] = rebalance_dates_v2(TrdDt, TrdDt[TrdDt.index(start_dt)-i], 
                                                                    TrdDt[TrdDt.index(end_dt)-i], freq_str)

        data_dict = {**backward_dict, 'Origin': Origin, **forward_dict}
        df = pd.DataFrame(data_dict)
    
    else:
        for i in range(roll_window_int + 1):
            assign_shift = []
            if i == 0:
                Origin = rebalance_dates_v2(TrdDt, start_dt, end_dt, freq_str, assign)
                continue
            for j in range(len(assign)):
                assign_shift.append(TrdDt[TrdDt.index(assign[j])+i])
            forward_dict[f'{i}-Day After'] = assign_shift
            
        data_dict = {'Origin': Origin, **forward_dict}
        df = pd.DataFrame(data_dict)
        return df
    
    return df.iloc[0,:],df.iloc[-1,:]


# In[15]:


def frame_df(factor_df, sec_name_df, trd_dt, TrdDt, start_dt, end_dt, freq_str = 5, assign = False, 
             num_grps_int=5, delay_int = 1):
        
    schedule = buy_sell_single(factor_df, TrdDt, start_dt, end_dt, freq_str,assign, num_grps_int, delay_int)

    # Create empty lists
    Sec_id = []
    Date = []
    Group = []

    # reb_dates = rebalance_dates_v2(TrdDt, start_dt, end_dt, freq_str)

    # Iterate over schedule columns and rows
    for i in range(len(schedule.columns)):
        for j in range(len(schedule.index)):
            for k in schedule.iloc[j, i]:
                Sec_id.append(sec_id(k, sec_name_df))
                Date.append(schedule.columns[i])
                Group.append(schedule.index[j])

        if i != (len(schedule.columns) - 1):
            # Extend lists using list repetition
            Sec_id.extend(Sec_id[-len(factor_df.columns) + 1:] * (days_diff(TrdDt, convert_dt(schedule.columns[i]),
                                                                           convert_dt(schedule.columns[i + 1])) - 1))
            Date.extend([num for num in date_diff(TrdDt, convert_dt(schedule.columns[i]),
                                                  convert_dt(schedule.columns[i + 1]))
                         for _ in range(len(factor_df.columns) - 1)])
            Group.extend(Group[-len(factor_df.columns) + 1:] * (days_diff(TrdDt, convert_dt(schedule.columns[i]),
                                                                         convert_dt(schedule.columns[i + 1])) - 1))

    Sec_id.extend(Sec_id[-len(factor_df.columns) + 1:] * (days_diff(TrdDt, convert_dt(schedule.columns[-1]), end_dt)))
    Date.extend([num for num in date_diff(TrdDt, convert_dt(schedule.columns[-1]), TrdDt[TrdDt.index(end_dt) + 1])
                  for _ in range(len(factor_df.columns) - 1)])
    Group.extend(Group[-len(factor_df.columns) + 1:] * (days_diff(TrdDt, convert_dt(schedule.columns[-1]), end_dt)))

    # Create a DataFrame using a dictionary
    frame_dict = {'Sec_id': Sec_id, 'Date': Date, 'Group': Group}
    frame = pd.DataFrame(frame_dict)

    # Use merge instead of multiple calls to loc for better performance
    frame = frame.merge(trd_dt[['IdxCd', 'TrdDt', 'ClsIdx']], how='left', left_on=['Sec_id', 'Date'], right_on=['IdxCd', 'TrdDt'])

    frame['SectorValue'] = None  # Create an empty column for sector values

    for i, row in frame.iterrows():
        date = row['Date']
        sector_id = row['Sec_id']
        sector_name = sec_name(sector_id, sec_name_df)  # Use the sec_name() function to get the sector name
     # Retrieve the corresponding sector value from factor_df DataFrame
        sector_value = factor_df.loc[factor_df['TrdDt'] == date, sector_name].values[0]

            # Assign the sector value to the 'SectorValue' column in frame DataFrame
        frame.at[i, 'SectorValue'] = sector_value
    
    return frame


# In[194]:


def back_test_v2(factor_df, sec_name_df, trd_dt, start_dt, end_dt, assign = False, rate = 0.002, freq_str = 5, 
                 num_grps_int=5, delay_int = 1, show_plots = True, show_table = True):
    
    TrdDt = TrdDates(factor_df)
    
    frame = frame_df(factor_df, sec_name_df, trd_dt, TrdDt, start_dt, end_dt, freq_str, assign, 
                     num_grps_int, delay_int)
    
    frame['TrdDt'] = pd.to_datetime(frame['TrdDt'])

    # Calculate individual sector returns
    frame['Return'] = frame.groupby('Sec_id')['ClsIdx'].pct_change()

    # Calculate cumulative returns per group
    grouped_df = frame.groupby(['Group','TrdDt'])['Return'].mean()
    df = grouped_df.reset_index()
    df['CumulativeReturn'] = df.groupby('Group')['Return'].cumsum()

    # Calculate relative cumulative percent change per group
    df2 = (df.groupby(['Group','TrdDt'])['Return'].mean() - grouped_df.groupby('TrdDt').mean()).reset_index()

    # Calculate the cumulative sum per group
    df2['CumulativeSum'] = df2.groupby('Group')['Return'].cumsum()

    max_group = len(frame['Group'].unique())
    sec_max_group = len(frame['Group'].unique()) - 1
    # Calculate the long/short relative return of Group 1-5
    group1_5_return = df2[df2['Group'] == 'Group1']['CumulativeSum'].values - df2[df2['Group'] == f'Group{max_group}']['CumulativeSum'].values

    # Calculate the long/short relative return of Group 2-4
    group2_4_return = df2[df2['Group'] == 'Group2']['CumulativeSum'].values - df2[df2['Group'] == f'Group{sec_max_group}']['CumulativeSum'].values

    reb_dates = rebalance_dates_v2(TrdDt, start_dt, end_dt, freq_str, assign)
    corr_series = []

    group1_sharpe_ratio = []
    group1_ExRet = []
    group1_Cal = []

    group1_5_ExRet = []
    group1_5_sharpe_ratio = []
    group1_5_Cal = []

    positive_count_ratio = []
    negative_count_ratio = []

    group1_diff_rate = []
    group5_diff_rate = []
    
    group1_max_drawdown = max_drawdown(df2[df2['Group'] == 'Group1']['CumulativeSum'])
    group1_5_max_drawdown = max_drawdown(pd.Series(group1_5_return))
    
    
    for j in range(len(reb_dates)-2):
        g1_diff_list = []
        g5_diff_list = []
        
        schedule_sectors = buy_sell_single(factor_df, TrdDt, reb_dates[j], reb_dates[j+2], freq_str, assign,
                                           num_grps_int, delay_int)
        compo_g1 = schedule_sectors.loc['Group1']
        compo_g5 = schedule_sectors.loc['Group5']
        for k in range(1, len(compo_g1)):
            g1_compo_prev = set(compo_g1[k-1])
            g1_compo_curr = set(compo_g1[k])
            g1_diff_rate_curr = len(g1_compo_curr - g1_compo_prev)/len(g1_compo_prev)
            g1_diff_rate_prev = len(g1_compo_prev - g1_compo_curr)/len(g1_compo_prev)
            g1_diff_list.append(g1_diff_rate_curr+g1_diff_rate_prev)
            
            g5_compo_prev = set(compo_g5[k-1])
            g5_compo_curr = set(compo_g5[k])
            g5_diff_rate_curr = len(g5_compo_curr - g5_compo_prev)/len(g5_compo_prev)
            g5_diff_rate_prev = len(g5_compo_prev - g5_compo_curr)/len(g5_compo_prev)
            g5_diff_list.append(g5_diff_rate_curr+g5_diff_rate_prev)
        
        group1_diff_rate.append(np.mean(g1_diff_list))
        group5_diff_rate.append(np.mean(g5_diff_list))
        
        # both sides' change rates BOTH with respect to the previous cycle
    
    for i in range(len(reb_dates)-1):
        
        Return_rank = []
        SectorValue_rank = []

        testdf2 = frame[frame['Date']==str(reb_dates[i])]
        SectorValue_rank = testdf2[['IdxCd','SectorValue']].sort_values('IdxCd').reset_index().iloc[:,-1].rank(ascending=False)

        subset = frame[(frame['Date'] > str(reb_dates[i])) & (frame['Date'] <= str(reb_dates[i+1]))]
        testdf = subset.groupby(['IdxCd', 'TrdDt'])['Return'].mean().reset_index()
        testdf['CumulativeReturn'] = testdf.groupby('IdxCd')['Return'].cumsum()
        Return_rank = testdf.pivot(columns='IdxCd', index='TrdDt', values='CumulativeReturn').iloc[-1, :].rank(ascending=False)

        correlation = stats.spearmanr(Return_rank, SectorValue_rank)[0]
        corr_series.append(correlation)

        # Calculate annualized Excess Return of Group 1
        subset2 = df2[(df2['TrdDt'] > str(reb_dates[i])) & (df2['TrdDt'] <= str(reb_dates[i+1]))]
        group1_excess_return = (subset2[subset2['Group'] == 'Group1']['Return'].cumsum().iloc[-1]) * (252 / len(subset2[subset2['Group'] == 'Group1']['Return']))
        group1_ExRet.append(group1_excess_return)

        # Calculate annualized Sharpe ratio of Group 1
        group1_std = np.std(subset2[subset2['Group'] == 'Group1']['Return']) * np.sqrt(252)
        group1_sharpe_ratio.append(group1_excess_return / group1_std)

        # Calculate Calmar ratio of Group 1
        group1_calmar_ratio = group1_excess_return / abs(group1_max_drawdown)
        group1_Cal.append(group1_calmar_ratio)

        # Calculate annulized Excess Return of Long/Short (Group 1 - Group 5)
        sub_group1_5_return = subset2[subset2['Group'] == 'Group1']['Return'].values - subset2[subset2['Group'] == f'Group{max_group}']['Return'].values
        group1_5_excess_return = sub_group1_5_return.cumsum()[-1] * (252 / len(subset2[subset2['Group'] == 'Group1']['Return']))
        group1_5_ExRet.append(group1_5_excess_return)

        # Calculate annualized Sharpe ratio of Long/Short (Group 1 - Group 5)
        g1_ret = subset2[subset2['Group'] == 'Group1']['Return']
        g5_ret = subset2[subset2['Group'] == 'Group5']['Return']
        group1_5_std = np.sqrt(np.var(g1_ret) + np.var(g5_ret)) * np.sqrt(252)
        group1_5_sharpe_ratio.append(group1_5_excess_return / group1_5_std)

        # Calculate Calmar ratio of Long/Short (Group 1 - Group 5)
        group1_5_calmar_ratio = group1_5_excess_return / abs(group1_max_drawdown)
        group1_5_Cal.append(group1_5_calmar_ratio)

        # Calculate Group1 Prediction Accuracy & Group5 Prediction Accuracy
        positive_count_group1 = 0
        total_count_group1 = 0

        group1_subset = subset2[subset2['Group'] == 'Group1'].copy()

        # Calculate the relative cumulative return
        group1_subset.loc[:, 'RelativeReturn'] = group1_subset.groupby('TrdDt')['Return'].cumsum()

        # Count the positive relative cumulative return sectors for Group 1
        positive_count_group1 = len(group1_subset[group1_subset['RelativeReturn'] > 0])
        total_count_group1 = len(group1_subset)

        # Calculate the positive count ratio for Group 1
        positive_count_ratio.append(positive_count_group1 / total_count_group1)

        # Negative count and total count for Group 5
        negative_count_group5 = 0
        total_count_group5 = 0

        group5_subset = subset2[subset2['Group'] == 'Group5'].copy()

        # Calculate the relative cumulative return
        group5_subset.loc[:, 'RelativeReturn'] = group5_subset.groupby('TrdDt')['Return'].cumsum()

        # Count the negative relative cumulative return sectors for Group 5
        negative_count_group5 = len(group5_subset[group5_subset['RelativeReturn'] < 0])
        total_count_group5 = len(group5_subset)

        # Calculate the negative count ratio for Group 5
        negative_count_ratio.append(negative_count_group5 / total_count_group5)
    
#     Return_rank = []
#     SectorValue_rank = []

#     testdf2 = frame[frame['Date']==str(reb_dates[-1])]
#     SectorValue_rank = testdf2[['IdxCd','SectorValue']].sort_values('IdxCd').reset_index().iloc[:,-1].rank(ascending=False)

#     subset = frame[(frame['Date'] > str(reb_dates[-1])) & (frame['Date'] <= str(end_dt))]
#     testdf = subset.groupby(['IdxCd', 'TrdDt'])['Return'].mean().reset_index()
#     testdf['CumulativeReturn'] = testdf.groupby('IdxCd')['Return'].cumsum()
#     Return_rank = testdf.pivot(columns='IdxCd', index='TrdDt', values='CumulativeReturn').iloc[-1, :].rank(ascending=False)
 
#     correlation = stats.spearmanr(Return_rank, SectorValue_rank)[0]
#     corr_series.append(correlation)

    # Convert corr_series to a pandas Series
    corr_series = pd.Series(corr_series)

    # Calculate the moving average using rolling
    window_size = 8  # Set the window size for the moving average
    moving_average = corr_series.rolling(window=window_size).mean()

    if show_plots:
        # Set up subplots
        fig, axs = plt.subplots(2, 2, figsize=(18, 8))

        # Plot absolute cumulative percent change per group
        axs[0, 0].set_title('Absolute Nav')
        groups = df['Group'].unique()
        for group in groups:
            group_df = df[df['Group'] == group]
            axs[0, 0].plot(group_df['TrdDt'], group_df['CumulativeReturn'], label=f'{group}')
        axs[0, 0].set_xlabel('Date')
        axs[0, 0].set_ylabel('Cumulative Percent Change')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot relative cumulative percent change per group
        axs[0, 1].set_title('Relative Nav')
        for group in groups:
            group_df = df2[df2['Group'] == group]
            axs[0, 1].plot(group_df['TrdDt'], group_df['CumulativeSum'], label=f'{group}', linestyle='-')
        axs[0, 1].set_xlabel('Date')
        axs[0, 1].set_ylabel('Relative Cumulative Percent Change')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot long/short relative return
        axs[1, 0].set_title('Long/Short Relative Return')
        axs[1, 0].plot(df2[df2['Group'] == 'Group1']['TrdDt'], group1_5_return, label='Long/Short 1')
        axs[1, 0].plot(df2[df2['Group'] == 'Group2']['TrdDt'], group2_4_return, label='Long/Short 2')
        axs[1, 0].set_xlabel('Date')
        axs[1, 0].set_ylabel('Relative Return')
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Plot the correlation series and moving average
#         axs[1, 0].vlines(reb_dates, ymin=0, ymax=corr_series)
#         axs[1, 0].plot(reb_dates, moving_average, label='Moving Average', color='red')
        axs[1, 1].vlines(reb_dates[:-1], ymin=0, ymax=corr_series)
        axs[1, 1].plot(reb_dates[:-1], moving_average, label='Moving Average', color='red')
        axs[1, 1].axhline(y=0, color='black', linestyle='--')
        axs[1, 1].set_xlabel('Date')
        axs[1, 1].set_ylabel('Rank IC')
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Adjust spacing between subplots
        plt.tight_layout()

        plt.show()

    if show_table:
        # Calculate the statistics table
        g1_Ex_ret = np.mean(group1_ExRet)
        g1_5_Ex_ret = np.mean(group1_5_ExRet)
        g1_diff_rate = np.mean(group1_diff_rate)
        g5_diff_rate = np.mean(group5_diff_rate)
        g1_fee = g1_diff_rate*252/freq_str *rate
        g5_fee = g5_diff_rate*252/freq_str *rate
        
        table = pd.DataFrame()
        if not assign:
            table['Freq'] = [freq_str]
        table['多头超额'] = [f"{np.round(g1_Ex_ret * 100, 2)}%"]
        table['多头_Sharpe'] = [np.round(np.mean(group1_sharpe_ratio),2)]
        table['多头_Calmar'] = [np.round(np.mean(group1_Cal),2)]
        table['多空收益'] = [f"{np.round(g1_5_Ex_ret * 100, 2)}%"]
        table['多空_Sharpe'] = [np.round(np.mean(group1_5_sharpe_ratio),2)]
        table['多空_Calmar'] = [np.round(np.mean(group1_5_Cal),2)]
        table['多头扣费'] = [f"{np.round((g1_Ex_ret - g1_fee) * 100, 2)}%"]
        table['多空扣费'] = [f"{np.round((g1_5_Ex_ret - (g1_fee+g5_fee)) * 100 , 2)}%"]
        
        
        table['rankIC'] = [f"{np.round(np.mean(corr_series) * 100, 2)}%"]
        table['多头胜率'] = [f"{np.round(np.mean(positive_count_ratio) * 100, 2)}%"]
        table['空头胜率'] = [f"{np.round(np.mean(negative_count_ratio) * 100, 2)}%"]
        table['多头换手'] = [f"{np.round(g1_diff_rate * 100, 2)}%"]
        table['空头换手'] = [f"{np.round(g5_diff_rate * 100 , 2)}%"]
        
        # Display the statistics table
        return table


# # Part II. Demonstration

# In[240]:


factor_df = SamSig_IE
sec_name_df = Sec_Name
trd_dt = Sec_Trd
start_dt = convert_dt('2016-05-10')
end_dt = convert_dt('2023-04-27')
freq_str = 5
freq_list = [5, 10, 20]


# ## Default Back-Testing Scence

# In[174]:


back_test_v2(factor_df, sec_name_df, trd_dt, start_dt, end_dt)


# ## Multi-Frequency Testing

# In[264]:


def multi_freq(factor_df, sec_name_df, trd_dt, start_dt, end_dt, freq_list,assign = False, 
               rate = 0.002, rolling = False, num_grps_int = 5,
              delay_int = 1,roll_window_int = 3):
    
    analyze_table = []
    
    if not rolling and assign:
        raise ValueError('No multi-frequency test for assigned dates')
        return
    
    elif not rolling and not assign:
        for freq in freq_list:
            analyze_df = back_test_v2(factor_df, sec_name_df, trd_dt, start_dt, end_dt, assign,
                                      rate, freq, num_grps_int,delay_int, show_plots = False)
            analyze_table.append(analyze_df)
        table = pd.concat(analyze_table, ignore_index=True)
    
    elif rolling and not assign:
        start_dates = []
        for freq in freq_list:
            start,end = roll_rebalance_dates(TrdDates(factor_df), start_dt, end_dt, freq, 
                                             roll_window_int = round(freq*0.6+1))
            start_dates.extend(start.index.values)
            for start_item, end_item in zip(start,end):
                analyze_df = back_test_v2(factor_df, sec_name_df, trd_dt, start_item, end_item, assign,
                                          rate, freq, num_grps_int, delay_int, show_plots = False)
                analyze_table.append(analyze_df)
    
        table = pd.concat(analyze_table, ignore_index=True)
        table.insert(0, 'Start Date', start_dates)
        
    elif rolling and assign:
        df = roll_rebalance_dates(TrdDt, start_dt, end_dt, freq_str,assign, roll_window_int)
        for i in range(len(df.columns)):
            analyze_df = back_test_v2(factor_df, sec_name_df, trd_dt, start_dt, end_dt, list(df.iloc[:,i]),
                                          rate,num_grps_int=num_grps_int, 
                                      delay_int = delay_int, show_plots = False)
            analyze_table.append(analyze_df)
        table = pd.concat(analyze_table, ignore_index=True)
        start_dates = list(df.columns.values)
        table.insert(0, 'Start Date', start_dates)
        
    return table


# ## Rolling-Readjustment Dates for Sensitivity Testing

# In[29]:


multi_freq(factor_df, sec_name_df, trd_dt, start_dt, end_dt,
           freq_list, rate = 0.002, rolling = True, assign = assign, num_grps_int = 5,
              delay_int = 1,roll_window_int = 10)


# In[177]:


multi_freq(factor_df, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# # Part III. Back-Testing Factors

# # 行业有效量价因子
# 
# # 1. 动量因子

# ## 1.1 简单动量 & 1.2 二阶动量
# 
# 取不同期限5日、10日、15日、20日、40日、60日、120日、180日、250日
# 
# back_test_v2(factor_df, sec_name_df, trd_dt, start_dt, end_dt, rate = 0.002, freq_str = 5, num_grps_int=5, delay_int = 1, show_plots = True, show_table = True):

# In[243]:


second_momentum = pd.read_csv('second_momentum_factor.csv')
back_test_v2(second_momentum, sec_name_df, trd_dt, start_dt, end_dt) 


# In[178]:


multi_freq(second_momentum, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# ## 1.3 动量期限差

# In[244]:


mmt_diff_factor = pd.read_csv('momentum_diff_factor.csv')
back_test_v2(mmt_diff_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[221]:


multi_freq(mmt_diff_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# # 2. 交易波动因子
# 
# ## 2.1 成交金额波动

# In[245]:


roll_ValTrd_factor = pd.read_csv('rolling_ValTrd_factor.csv')
back_test_v2(roll_ValTrd_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[223]:


multi_freq(roll_ValTrd_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# ## 2.2 成交量波动

# In[246]:


roll_ShrTrd_factor = pd.read_csv('rolling_ShrTrd_factor.csv')
back_test_v2(roll_ShrTrd_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[225]:


multi_freq(roll_ShrTrd_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# # 3. 换手率因子
# 
# ## 3.1 换手率变化

# In[247]:


ex_rate_factor = pd.read_csv('ex_rate_factor.csv')
back_test_v2(ex_rate_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[227]:


multi_freq(ex_rate_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# # 4. 多空对比因子
# 
# ## 4.1 多空对比总量

# In[248]:


ls_diff_factor = pd.read_csv('ls_diff_factor.csv')
back_test_v2(ls_diff_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[229]:


multi_freq(ls_diff_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# ## 4.2 多空对比变化

# In[249]:


ls_diff_chg_factor = pd.read_csv('ls_diff_chg_factor.csv')
back_test_v2(ls_diff_chg_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[231]:


multi_freq(ls_diff_chg_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# # 5. 量价背离因子
# 
# ## 5.1 量价排序协方差

# In[250]:


cov_matrix_factor = pd.read_csv('cov_matrix_factor.csv')
back_test_v2(cov_matrix_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[251]:


multi_freq(cov_matrix_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# ## 5.2 量价相关系数

# In[252]:


corr_factor = pd.read_csv('corr_factor.csv')
back_test_v2(corr_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[253]:


multi_freq(corr_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# ## 5.3 一阶量价背离

# In[254]:


rank_corr_factor = pd.read_csv('rank_corr_factor.csv')
back_test_v2(rank_corr_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[255]:


multi_freq(rank_corr_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# # 6. 量幅同向因子

# In[256]:


rank_corr_price_factor = pd.read_csv('rank_corr_price_factor.csv')
back_test_v2(rank_corr_price_factor, sec_name_df, trd_dt, start_dt, end_dt) 


# In[257]:


multi_freq(rank_corr_price_factor, sec_name_df, trd_dt, start_dt, end_dt,freq_list, rolling = True)


# # 剩余模型收益因子
# 
# # 1. 财务变量
# 使用整体法直接对行业计算 
# 
# 账面市值比BM、销售净利率MOS、总资产周转率ATO、权益乘数EM
# 
# 由于上市公司财务数据更新频率较低，仅取每年4月30日、8月31日、和10月31日的数据。（公告日截止）
# 
# ## 1.1 账面市值比

# In[258]:


def assign_trddt(TrdDt, assign_list):
    '''
    Turns adjustment dates (assigned manually) into trading schedule
    '''

    if type(assign_list[0]) != dt.date:
        assign_list = [convert_dt(date_str) for date_str in assign_list]

    def find_next_trading_day(start_date):
        delta = dt.timedelta(days=1)
        while start_date not in TrdDt:
            start_date += delta
        return start_date

    result_list = []

    if assign_list[0] not in TrdDt:
        # If the first element is not in TrdDt, find the next closest trading day
        result_list.append(find_next_trading_day(assign_list[0]))
    else:
        result_list.append(assign_list[0])

    for i in range(1, len(assign_list)):
        prev_date = result_list[-1]
        current_date = assign_list[i]

        if current_date == prev_date:
            # If the current date is the same as the previous one, increment by 1 day
            current_date += dt.timedelta(days=1)

        while current_date not in TrdDt:
            # Find the next closest trading day
            current_date = find_next_trading_day(current_date + dt.timedelta(days=1))

        result_list.append(current_date)

    return result_list


# In[294]:


assign_str = ['2011-05-01','2011-09-01','2011-11-01','2012-05-01','2012-09-01','2012-11-01',
          '2013-05-01','2013-09-01','2013-11-01','2014-05-01','2014-09-01','2014-11-01',
          '2015-05-01','2015-09-01','2015-11-01','2016-05-01','2016-09-01','2016-11-01',
          '2017-05-01','2017-09-01','2017-11-01','2018-05-01','2018-09-01','2018-11-01',
          '2019-05-01','2019-09-01','2019-11-01','2020-05-01','2020-09-01','2020-11-01',
          '2021-05-01','2021-09-01','2021-11-01','2022-05-01','2022-09-01','2022-11-01']
TrdDt = TrdDates(mean_BM_ratio_factor)
assign = assign_trddt(TrdDt, assign_str)


# 输入变量原始值无法预测行业走势

# In[297]:


mean_BM_ratio_factor = pd.read_csv('filtered_test_df.csv').iloc[:,1:]
back_test_v2(mean_BM_ratio_factor, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 0, show_plots = True, show_table = True)


# In[266]:


multi_freq(mean_BM_ratio_factor, sec_name_df, trd_dt, assign[0], assign[-1],
           freq_list, rate = 0.002, rolling = True, assign = assign, num_grps_int = 5,
              delay_int = 1,roll_window_int = 10)


# 输入一阶变量(季度窗口)变动值
# 
# 

# In[283]:


TrdDt = TrdDates(filtered_expanded_lag_mean_BM_ratio)
diff_assign_str = ['2011-09-01','2011-11-01','2012-05-01','2012-09-01','2012-11-01',
          '2013-05-01','2013-09-01','2013-11-01','2014-05-01','2014-09-01','2014-11-01',
          '2015-05-01','2015-09-01','2015-11-01','2016-05-01','2016-09-01','2016-11-01',
          '2017-05-01','2017-09-01','2017-11-01','2018-05-01','2018-09-01','2018-11-01',
          '2019-05-01','2019-09-01','2019-11-01','2020-05-01','2020-09-01','2020-11-01',
          '2021-05-01','2021-09-01','2021-11-01','2022-05-01','2022-09-01','2022-11-01']

assign = assign_trddt(TrdDt, diff_assign_str)


# In[284]:


filtered_expanded_lag_mean_BM_ratio = pd.read_csv('filtered_expanded_lag_mean_BM_ratio.csv').iloc[:,1:]
back_test_v2(filtered_expanded_lag_mean_BM_ratio, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 1, show_plots = True, show_table = True)


# In[268]:


multi_freq(filtered_expanded_lag_mean_BM_ratio, sec_name_df, trd_dt, assign[0], assign[-1],
           freq_list, rate = 0.002, rolling = True, assign = assign, num_grps_int = 5,
              delay_int = 1,roll_window_int = 10)


# ## 1.2 销售净利率

# 输入变量原始值2021年末开始无法预测行业走势 

# In[269]:


filtered_mean_MOS_ratio = pd.read_csv('filtered_mean_MOS_ratio.csv').iloc[:,1:]

assign = assign_trddt(TrdDates(filtered_mean_MOS_ratio), assign_str)

back_test_v2(filtered_mean_MOS_ratio, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 1, show_plots = True, show_table = True)


# In[273]:


TrdDt = TrdDates(filtered_mean_MOS_ratio)
assign = assign_trddt(TrdDt, assign_str)

multi_freq(filtered_mean_MOS_ratio, sec_name_df, trd_dt, assign[0], assign[-1],
           freq_list, rate = 0.002, rolling = True, assign = assign, num_grps_int = 5,
              delay_int = 1,roll_window_int = 10)


# 输入一阶变量(季度窗口)变动值 
# 2020年后失去预测能力

# In[274]:


filtered_expanded_lag_mean_MOS_ratio = pd.read_csv('filtered_expanded_lag_mean_MOS_ratio.csv').iloc[:,1:]

assign = assign_trddt(TrdDates(filtered_expanded_lag_mean_MOS_ratio), diff_assign_str)

back_test_v2(filtered_expanded_lag_mean_MOS_ratio, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 1, show_plots = True, show_table = True)


# In[275]:


TrdDt = TrdDates(filtered_expanded_lag_mean_MOS_ratio)
assign = assign_trddt(TrdDt, diff_assign_str)
multi_freq(filtered_expanded_lag_mean_MOS_ratio, sec_name_df, trd_dt, assign[0], assign[-1],
           freq_list, rate = 0.002, rolling = True, assign = assign, num_grps_int = 5,
              delay_int = 1,roll_window_int = 10)


# ## 1.3 总资产周转率

# 输入变量原始值预测行业走势

# In[281]:


filtered_mean_ATO_ratio = pd.read_csv('filtered_mean_ATO_ratio.csv').iloc[:,1:]

assign = assign_trddt(TrdDates(filtered_mean_ATO_ratio), assign_str)

back_test_v2(filtered_mean_ATO_ratio, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 1, show_plots = True, show_table = True)


# In[276]:


TrdDt = TrdDates(filtered_mean_ATO_ratio)
assign = assign_trddt(TrdDt, assign_str)

multi_freq(filtered_mean_ATO_ratio, sec_name_df, trd_dt, assign[0], assign[-1],
           freq_list, rate = 0.002, rolling = True, assign = assign, num_grps_int = 5,
              delay_int = 1,roll_window_int = 10)


# 输入一阶变量(季度窗口)变动值 
# 
# 不论正反，效果并不明显，且如相反数与逻辑相违背

# In[280]:


filtered_expanded_lag_mean_ATO_ratio = pd.read_csv('filtered_expanded_lag_mean_ATO_ratio.csv').iloc[:,1:]

assign = assign_trddt(TrdDates(filtered_expanded_lag_mean_ATO_ratio), diff_assign_str)

back_test_v2(filtered_expanded_lag_mean_ATO_ratio, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 1, show_plots = True, show_table = True)


# In[277]:


TrdDt = TrdDates(filtered_expanded_lag_mean_ATO_ratio)
assign = assign_trddt(TrdDt, diff_assign_str)
multi_freq(filtered_expanded_lag_mean_ATO_ratio, sec_name_df, trd_dt, assign[0], assign[-1],
           freq_list, rate = 0.002, rolling = True, assign = assign, num_grps_int = 5,
              delay_int = 1,roll_window_int = 10)


# ## 1.4 权益乘数
# 
# 输入变量原始值预测行业走势
# 
# 即EM越大得分越低

# In[218]:


filtered_mean_EM_ratio = pd.read_csv('filtered_mean_EM_ratio.csv').iloc[:,1:]

assign = assign_trddt(TrdDates(filtered_mean_EM_ratio), assign_str)

back_test_v2(filtered_mean_EM_ratio, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 1, show_plots = True, show_table = True)


# In[278]:


TrdDt = TrdDates(filtered_mean_EM_ratio)
assign = assign_trddt(TrdDt, assign_str)

multi_freq(filtered_mean_EM_ratio, sec_name_df, trd_dt, assign[0], assign[-1],
           freq_list, rate = 0.002, rolling = True, assign = assign, num_grps_int = 5,
              delay_int = 1,roll_window_int = 10)


# 输入一阶变量(季度窗口)变动值
# 
# 效果并不明显

# In[219]:


filtered_expanded_lag_mean_EM_ratio = pd.read_csv('filtered_expanded_lag_mean_EM_ratio.csv').iloc[:,1:]

assign = assign_trddt(TrdDates(filtered_expanded_lag_mean_EM_ratio), diff_assign_str)

back_test_v2(filtered_expanded_lag_mean_EM_ratio, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 1, show_plots = True, show_table = True)


# In[279]:


TrdDt = TrdDates(filtered_expanded_lag_mean_EM_ratio)
assign = assign_trddt(TrdDt, diff_assign_str)
multi_freq(filtered_expanded_lag_mean_EM_ratio, sec_name_df, trd_dt, assign[0], assign[-1],
           freq_list, rate = 0.002, rolling = True, assign = assign, num_grps_int = 5,
              delay_int = 1,roll_window_int = 10)


# ## 1.5 股权成本

# In[ ]:





# In[ ]:





# ## 等权复合
# ## Method I

# In[300]:


syn_fin_factor = pd.read_csv('syn_fin_factor.csv').iloc[:,1:]

assign = assign_trddt(TrdDates(syn_fin_factor), diff_assign_str)

back_test_v2(syn_fin_factor, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 1, show_plots = True, show_table = True)


# ## Method II

# In[304]:


syn_fin_factor_v2 = pd.read_csv('syn_fin_factor_v2.csv').iloc[:,1:]
assign = assign_trddt(TrdDates(syn_fin_factor_v2), diff_assign_str)

back_test_v2(syn_fin_factor_v2, sec_name_df, trd_dt, assign[0], assign[-1], assign = assign, rate = 0.002, 
             num_grps_int=5, delay_int = 1, show_plots = True, show_table = True)

