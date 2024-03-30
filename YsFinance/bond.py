import numpy as np

class CashFlow:
    
    def __init__(self) -> None:
        pass
    
    def i_trans_to_real(self,i,m):
        return (1+i/m)*m - 1
    
    def i_trans_to_nomi(self,i,m):
        return m*((1+i)**(1/m)-1)
    
    def a(self,n,i):
        nu = 1/(1+i)
        return (1-nu**n)/i
    
    def s(self,n,i):
        return ((1+i)^n-1)/i
    
    def a_dot(self,n,i):
        d = i/(1+i)
        nu = 1/(1+i)
        return (1-nu**n)/d
    
    def s_dot(self,n,i):
        d = i/(1+i)
        return ((1+i)**n-1)/d
    
    def a_inf(self,i):
        return 1/i
    
    def a_dot_inf(self,i):
        d = i/(1+i)
        return 1/d
    
    def Ia(self,n,i):
        nu = 1/(1+i)
        return (self.a_dot(n,i)-n*nu**n)/i
    
    def Is(self,n,i):
        return (self.s_dot(n,i)-n)/i
    
    def Da(self,n,i):
        return (n-self.a(n,i))/i
    
    def Ds(self,n,i):
        return (n*(1+i)**n-self.s(n,i))/i
    
    def Ia_dot(self,n,i):
        nu = 1/(1+i)
        d = i/(1+i)
        return (self.a_dot(n,i)-n*nu**n)/d
    
    def Is_dot(self,n,i):
        d = i/(1+i)
        return (self.s_dot(n,i)-n)/d
    
    def Da_dot(self,n,i):
        d = i/(1+i)
        return (n-self.a(n,i))/d
    
    def Ds_dot(self,n,i):
        d = i/(1+i)
        return (n*(1+i)**n-self.s(n,i))/d
    
    def a_rate_increase(self,n,i,k):
        if i == k:
            return n/(1+i)
        if i < k:
            raise ValueError("k > i then the value is infty")
        num = 1-((1+k)/(1+i))**n
        den = i-k
        return num/den
    
    def current_value(self,time_flow,cash_flow,i):
        if len(time_flow) != len(cash_flow):
            raise ValueError("length of time and cash should map.")
        
        time_flow = np.array(time_flow)
        cash_flow = np.array(cash_flow)
        return ((1+i)**(-time_flow) * cash_flow).sum()
    
    def current_duration(self,time_flow,cash_flow,i):
        if len(time_flow) != len(cash_flow):
            raise ValueError("length of time and cash should map.")
        time_flow = np.array(time_flow)
        cash_flow = np.array(cash_flow)
        price = self.current_value(time_flow,cash_flow,i)
        nu = 1/(1+i)
        return (time_flow * cash_flow * nu**(time_flow)/price).sum()
    
    def current_convexity(self,time_flow,cash_flow,i):
        if len(time_flow) != len(cash_flow):
            raise ValueError("length of time and cash should map.")
        time_flow = np.array(time_flow)
        cash_flow = np.array(cash_flow)
        price = self.current_value(time_flow,cash_flow,i)
        nu = 1/(1+i)
        return (time_flow**2 * cash_flow * nu**(time_flow)/price).sum()

    
cf = CashFlow()   



class BondValue:

    def __init__(self,F,T,i,r,n,C=None):
        """calculate the bond value.

        Args:
            F (float): 债券面值
            T (int): 剩余年限/年 
            i (float): 到期收益率/年
            r (float): 票面利率/年
            n (int): 每年付息次数
            C (float): 兑现值, 一般等于F
        """
        self.F = F
        self.T = T
        self.n = n
        self.i = i
        self.r = r
        if C is None:
            self.C = F
        else:
            self.C = C
        
    def current_price(self):
        time_flow = np.linspace(0,self.T,self.n*self.T+1)[1:]
        cash_flow = self.F*self.r/self.n*np.ones(self.T*self.n)
        cash_flow[-1] = cash_flow[-1]+self.C
        return cf.current_value(time_flow,cash_flow,self.i)
    
    def current_duration(self):
        price = self.current_price()
        time_flow = np.linspace(0,self.T,self.n*self.T+1)[1:]
        cash_flow = self.F*self.r/self.n*np.ones(self.T*self.n)
        cash_flow[-1] = cash_flow[-1]+self.C
        nu = 1/(1+self.i)
        return (time_flow * cash_flow * nu**(time_flow)/price).sum()
        
    
    def book_value(self,t):
        ts = np.linspace(0,self.T,self.n*self.T+1)
        gap = 1/self.n
        loc = int(t/gap)
        pass
        