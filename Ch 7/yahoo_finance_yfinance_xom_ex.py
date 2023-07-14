import yfinance as yf

xom = yf.Ticker("xom")
xom

xom_historical = xom.history(start="2020-06-01", end="2020-10-12", interval="1wk")
xom_historical

xom_data = yf.download("xom", start="2005-12-01", end="2005-12-10", group_by='tickers')
xom_data

type(xom_data)

djia = yf.Ticker("^DJI")
djia

djia_data = yf.download("^DJI", start="2005-12-01", end="2005-12-10", group_by='tickers')
djia_data