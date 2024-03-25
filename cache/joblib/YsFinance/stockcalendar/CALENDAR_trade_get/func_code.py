# first line: 15
@memory.cache
def CALENDAR_trade_get():
    df = pro.trade_cal(exchange='',start_date=STARTDATE,end_date=ENDDATE)
    df.columns = ['exchange','cal_date','is_open','pretrade_date']
    df.set_index('cal_date',inplace=True)
    df.index = pd.to_datetime(df.index)
    df['pretrade_date'] = pd.to_datetime(df['pretrade_date'])
    df.sort_index(inplace=True)
    return df
