
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:34:34 2021

@author: aidac
"""
# from IPython import get_ipython
# get_ipython().magic('reset -sf')    #löscht alle Variablen
# get_ipython().magic('clear')        # löscht alte Consolen Ausgaben

#from numpy import genfromtxt

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from datetime import date
import requests
from scipy import stats
from scipy.stats import norm,t,chi2,f
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker



def conf_pred_band_ex(_regress_ex, _poly, _model, alpha=0.05):
    """ Function calculates the confidence and prediction interval for a
    given multivariate regression function poly according to lecture DFSS,
    regression parameters are already determined in an existing model,
    identical polynom is used for extrapolation

    Parameters
    ----------
    regress_ex : DataFrame
        Extended dataset for calculation.
    poly : OLS object of statsmodels.regression.linear_model modul
        definition of regression model.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Model parameters.
    alpha : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    lconf_ex : Series
        Distance of confidence limit to mean regression.
    lpred_ex : Series
        Distance of prediction limit to mean regression.
    """

    # ols is used to calculte the complets vector x_0 of input variables
    poly_ex = ols(_poly.formula, _regress_ex)
    x_0 = poly_ex.exog
    # Calculation according lecture book
    d = np.dot(x_0, np.dot(_poly.normalized_cov_params, x_0.T))
    c_1 = stats.t.isf(alpha/2, _model.df_resid)
    lconf_ex = c_1*np.sqrt(np.diag(d)*_model.mse_resid)
    lpred_ex = c_1*np.sqrt((1+np.diag(d))*_model.mse_resid)

    return lconf_ex, lpred_ex

# Checken auf neue Daten, dazu erst alte Daten auslesen und Datum speichern



headers=['date','dosen_kumulativ','dosen_differenz_zum_vortag','dosen_erst_differenz_zum_vortag','dosen_zweit_differenz_zum_vortag','dosen_biontech_kumulativ','dosen_moderna_kumulativ','dosen_astrazeneca_kumulativ','personen_erst_kumulativ','personen_voll_kumulativ','impf_quote_erst','impf_quote_voll','indikation_alter_dosen','indikation_beruf_dosen','indikation_medizinisch_dosen','indikation_pflegeheim_dosen','indikation_alter_erst','indikation_beruf_erst','indikation_medizinisch_erst','indikation_pflegeheim_erst','indikation_alter_voll','indikation_beruf_voll','indikation_medizinisch_voll','indikation_pflegeheim_voll','dosen_dim_kumulativ','dosen_kbv_kumulativ','dosen_johnson_kumulativ','dosen_biontech_erst_kumulativ','dosen_biontech_zweit_kumulativ','dosen_moderna_erst_kumulativ','dosen_moderna_zweit_kumulativ','dosen_astrazeneca_erst_kumulativ','dosen_astrazeneca_zweit_kumulativ','dosen_erst_kumulativ','dosen_zweit_kumulativ']

data=pd.read_csv('downloaded.csv',sep='	',names=headers,header=0)

today=date.today()

""" Entscheidung ob Test auf neue Daten oder nicht """
last_day_data= today.strftime("%m_%d_%Y")
#last_day_data=data.date[len(data)-1]

header_supply=['date','impfstoff','region','dosen','einrichtung']
data_supply=pd.read_csv('downloaded_lieferung.csv',sep='	',names=header_supply,header=0)
last_day_supply=data_supply.date[len(data_supply)-1]


""" Neue Daten herunterladen"""

impfung_tsv_url='https://impfdashboard.de/static/data/germany_vaccinations_timeseries_v2.tsv'

req = requests.get(impfung_tsv_url)
url_content = req.content
csv_file = open('downloaded.csv', 'wb')
csv_file.write(url_content)
csv_file.close()


lieferung_tsv_url='https://impfdashboard.de/static/data/germany_deliveries_timeseries_v2.tsv'

req = requests.get(lieferung_tsv_url)
url_content = req.content
csv_file = open('downloaded_lieferung.csv', 'wb')
csv_file.write(url_content)
csv_file.close()




data=pd.read_csv('downloaded.csv',sep='	',names=headers,header=0)


header_supply=['date','impfstoff','region','dosen','einrichtung']
data_supply=pd.read_csv('downloaded_lieferung.csv',sep='	',names=header_supply,header=0)

if ((data.date[len(data)-1]==last_day_data) and (data_supply.date[len(data_supply)-1]==last_day_supply)):
    print("Keine neuen Daten")
else:
    data_supply = data_supply.sort_values(by=["date","impfstoff"])
    data_supply['dosen_gesamt']=0
    data_supply['dosen_kummulativ']=0
    data_supply['biontech']=0
    data_supply['biontech_gesamt']=0
    data_supply['moderna']=0
    data_supply['moderna_gesamt']=0
    data_supply['astra']=0
    data_supply['astra_gesamt']=0
    data_supply['johnson']=0
    data_supply['johnson_gesamt']=0
    
    
    data['mean_weekly']=0
    
    days=len(data)
    current_day=days-1
    print("Es liegen die Daten vom: ",data.date[current_day],"vor.")
    data['days']=np.linspace(1,days,days)
    df_date=data[['date','days']]
    df_date_concat=pd.DataFrame()
    
    forecast=5
    df_date_concat['date']=pd.date_range(df_date.date[current_day],periods=forecast,freq='D').strftime("%Y-%m-%d")
    df_date_concat['days']=np.arange(current_day+1,current_day+1+forecast,1)
    df_date=pd.concat([df_date,df_date_concat])
    
    
    
    len_supply=len(data_supply)
    cur_day=data_supply.date[0]
    day_sum=0
    
    biontech_day_sum=0
    astra_day_sum=0
    moderna_day_sum=0
    johnson_day_sum=0
    
    kum_sum=0
    biontech_kum_sum=0
    astra_kum_sum=0
    moderna_kum_sum=0
    johnson_kum_sum=0
    
    i=0
    last_i=i
    data_dates=data.date.unique()   
    dates=data_supply.date.unique()
    
    for i in dates:
    #if i==cur_day:
        day_sum=data_supply.loc[data_supply.date==i].dosen.sum()
        biontech_day_sum=data_supply.loc[((data_supply.date==i)&(data_supply.impfstoff=='comirnaty'))].dosen.sum()
        moderna_day_sum=data_supply.loc[((data_supply.date==i)&(data_supply.impfstoff=='moderna'))].dosen.sum()
        astra_day_sum=data_supply.loc[((data_supply.date==i)&(data_supply.impfstoff=='astra'))].dosen.sum()
        johnson_day_sum=data_supply.loc[((data_supply.date==i)&(data_supply.impfstoff=='johnson'))].dosen.sum()
        
        
        # if data_supply.impfstoff[i] == 'comirnaty':
        #     biontech_day_sum=biontech_day_sum+data_supply.dosen[i]
        # elif data_supply.impfstoff[i] == 'moderna':
        #     moderna_day_sum=moderna_day_sum+data_supply.dosen[i]
        # elif data_supply.impfstoff[i] == 'astra':
        #     astra_day_sum=astra_day_sum+data_supply.dosen[i]
        # elif data_supply.impfstoff[i] == 'johnson':
        #     johnson_day_sum=johnson_day_sum+data_supply.dosen[i]
    #else:
        kum_sum=kum_sum+day_sum
        biontech_kum_sum=biontech_kum_sum+biontech_day_sum
        astra_kum_sum=astra_kum_sum+astra_day_sum
        moderna_kum_sum=moderna_kum_sum+moderna_day_sum
        johnson_kum_sum=johnson_kum_sum+johnson_day_sum
        
        
        data_supply.loc[(data_supply.date==i),'dosen_gesamt']=day_sum
        data_supply.loc[(data_supply.date==i),'dosen_kummulativ']=kum_sum
        data_supply.loc[(data_supply.date==i),'biontech']=biontech_day_sum
        data_supply.loc[(data_supply.date==i),'astra']=astra_day_sum
        data_supply.loc[(data_supply.date==i),'moderna']=moderna_day_sum
        data_supply.loc[(data_supply.date==i),'johnson']=johnson_day_sum
       
        data_supply.loc[(data_supply.date==i),'biontech_gesamt']=biontech_kum_sum
        data_supply.loc[(data_supply.date==i),'astra_gesamt']=astra_kum_sum
        data_supply.loc[(data_supply.date==i),'moderna_gesamt']=moderna_kum_sum
        data_supply.loc[(data_supply.date==i),'johnson_gesamt']=johnson_kum_sum
        
        
        #for j in range(last_i,i):
            # data_supply.loc[j,'dosen_gesamt']=day_sum
            # data_supply.loc[j,'dosen_kummulativ']=kum_sum
            # data_supply.loc[j,'biontech']=biontech_day_sum
            # data_supply.loc[j,'astra']=astra_day_sum
            # data_supply.loc[j,'moderna']=moderna_day_sum
            # data_supply.loc[j,'johnson']=johnson_day_sum
            
            # data_supply.loc[j,'biontech_gesamt']=biontech_kum_sum
            # data_supply.loc[j,'astra_gesamt']=astra_kum_sum
            # data_supply.loc[j,'moderna_gesamt']=moderna_kum_sum
            # data_supply.loc[j,'johnson_gesamt']=johnson_kum_sum
            
        # cur_day=data_supply.date[i]
        # day_sum=data_supply.dosen[i]
        # last_i=i
        # if data_supply.impfstoff[i] == 'comirnaty':
        #     biontech_day_sum=data_supply.dosen[i]
        #     astra_day_sum=0
        #     moderna_day_sum=0
        #     johnson_day_sum=0
        # elif data_supply.impfstoff[i] == 'moderna':
        #     moderna_day_sum=data_supply.dosen[i]
        #     biontech_day_sum=0
        #     astra_day_sum=0
        #     johnson_day_sum=0
        # elif data_supply.impfstoff[i] == 'astra':
        #     astra_day_sum=data_supply.dosen[i]
        #     biontech_day_sum=0
        #     moderna_day_sum=0
        #     johnson_day_sum=0
        # elif data_supply.impfstoff[i] == 'johnson':
        #     johnson_day_sum=data_supply.dosen[i]
        #     astra_day_sum=0
        #     biontech_day_sum=0
        #     moderna_day_sum=0
  
    
    
    # kum_sum=kum_sum+day_sum
    # biontech_kum_sum=biontech_kum_sum+biontech_day_sum
    # astra_kum_sum=astra_kum_sum+astra_day_sum
    # moderna_kum_sum=moderna_kum_sum+moderna_day_sum
    # johnson_kum_sum=johnson_kum_sum+johnson_day_sum
    
    # for j in range(last_i,i+1):
    #     data_supply.loc[j,'dosen_gesamt']=day_sum
    #     data_supply.loc[j,'dosen_kummulativ']=kum_sum
    #     data_supply.loc[j,'biontech']=biontech_day_sum
    #     data_supply.loc[j,'astra']=astra_day_sum
    #     data_supply.loc[j,'moderna']=moderna_day_sum
    #     data_supply.loc[j,'johnson']=johnson_day_sum
    #     data_supply.loc[j,'biontech_gesamt']=biontech_kum_sum
    #     data_supply.loc[j,'astra_gesamt']=astra_kum_sum
    #     data_supply.loc[j,'moderna_gesamt']=moderna_kum_sum
    #     data_supply.loc[j,'johnson_gesamt']=johnson_kum_sum
        
    kum_sum_check=biontech_kum_sum+astra_kum_sum+moderna_kum_sum+johnson_kum_sum
    if kum_sum== kum_sum_check:
        print('Die Berechnung war richtig')
    else:
        print('Die Differenz beträgt:',(kum_sum-kum_sum_check))
    
    """Wöchentliche Mittelwerte"""
    weeks=int(days/7)
    rest_days=days%7-1
    
    for i in range(weeks):
        curr_mean=np.mean(data.dosen_differenz_zum_vortag[(weeks-i)*7-6+rest_days:(weeks-i)*7+rest_days])
        for j in range(7):
            data.loc[(weeks-i)*7-j+rest_days,'mean_weekly']=curr_mean
    for i in range(rest_days+1):
        curr_mean=np.mean(data.dosen_differenz_zum_vortag[:rest_days+1])
        for j in range(rest_days+1):
            data.loc[j,'mean_weekly']=curr_mean
    
    poly = ols("mean_weekly ~ 0+I(days**2)+I(days*3)+I(days**5)", data)
    model = poly.fit()
    b=model.params
    #print(model.summary())
    
    GAMMA = 0.95
    
    """Experiment"""
    prediction_plot = pd.DataFrame({"days": np.linspace(1, days+forecast, days+forecast)})
    prediction_plot["mean_weekly"] = model.predict(prediction_plot)
    #prediction_plot["confidence"], prediction_plot["prediction"] = \
    #    conf_pred_band_ex(prediction_plot, poly, model, alpha=1-GAMMA)
    
    
    """ Berechnung der aktuell gelagerten Dosen"""
    biontech_storage_p=data.dosen_biontech_kumulativ[days-1]/max(data_supply.biontech_gesamt)*100;
    astra_storage_p=data.dosen_astrazeneca_kumulativ[days-1]/max(data_supply.astra_gesamt)*100;
    moderna_storage_p=data.dosen_moderna_kumulativ[days-1]/max(data_supply.moderna_gesamt)*100;
    johnson_storage_p=data.dosen_johnson_kumulativ[days-1]/max(data_supply.johnson_gesamt)*100;
    
    biontech_storage=np.abs(data.dosen_biontech_kumulativ[days-1]-max(data_supply.biontech_gesamt))
    astra_storage=np.abs(data.dosen_astrazeneca_kumulativ[days-1]-max(data_supply.astra_gesamt))
    moderna_storage=np.abs(data.dosen_moderna_kumulativ[days-1]-max(data_supply.moderna_gesamt))
    johnson_storage=np.abs(data.dosen_johnson_kumulativ[days-1]-max(data_supply.johnson_gesamt))
    
    blue_plot='#1f77b4'
    
    fig1=plt.figure(1,figsize=(10,25))
    
    gs = GridSpec(4, 2, figure=fig1)
    
    fig1.suptitle("Impfstatistik mit Daten vom: "+data.date[current_day], fontsize=24)
        
    
    pie_label=['Keine Impfung','Erstimpfung','Vollständige Impfung']
    pie_values=[100-np.round(data.impf_quote_erst[current_day]*100,1),\
                np.round(data.impf_quote_erst[current_day]*100-data.impf_quote_voll[current_day]*100,1),np.round(data.impf_quote_voll[current_day]*100,1)]
    explode = (0, 0.1, 0.1,)
    
    
    """Pie Chart"""
    ax1 = fig1.add_subplot(gs[0, 0])
    ax1.pie(pie_values,labels=pie_label,autopct='%0.1f%%',shadow=True,startangle=90,explode=explode)
    ax1.set_title('Impffortschritt der Gesamtbevölkerung')
    
    
    """Hersteller Lagerbestand Chart"""
    ax2 = fig1.add_subplot(gs[0, 1])
    ax2.bar(x=['Biontech','Astrazeneca','Moderna','Johnson'],\
                height=[max(data_supply.biontech_gesamt)/1e6,\
                    max(data_supply.astra_gesamt)/1e6,\
                    max(data_supply.moderna_gesamt)/1e6,\
                    max(data_supply.johnson_gesamt)/1e6],\
                    color='gray',width=0.4,\
                    label="Gelieferte Dosen")
    ax2.bar(x=['Biontech','Astrazeneca','Moderna','Johnson'],\
                height=[data.dosen_biontech_kumulativ[days-1]/1e6,\
                    data.dosen_astrazeneca_kumulativ[days-1]/1e6,\
                    data.dosen_moderna_kumulativ[days-1]/1e6,\
                    data.dosen_johnson_kumulativ[days-1]/1e6],\
                    width=0.4,\
                    color=['darkblue','darkgreen','darkorange','r'],\
                    alpha=0.9,label="Verabreichte Dosen")
       
    
    str_biontech_storag="{}%".format(round(biontech_storage_p,2))    
    str_astra_storag="{}%".format(round(astra_storage_p,2))    
    str_moderna_storag="{}%".format(round(moderna_storage_p,2))    
    str_johnson_storag="{}%".format(round(johnson_storage_p,2))    
    
    
    ax2.text(0.25,data.dosen_biontech_kumulativ[days-1]/1e6,str_biontech_storag,color='darkblue')
    ax2.text(1.25,data.dosen_astrazeneca_kumulativ[days-1]/1e6,str_astra_storag,color='darkgreen')
    ax2.text(2.25,data.dosen_moderna_kumulativ[days-1]/1e6,str_moderna_storag,color='darkorange')
    ax2.text(3.25,data.dosen_johnson_kumulativ[days-1]/1e6,str_johnson_storag,color='r')
    ax2.grid(True)
    ax2.legend(['Gelieferte Dosen','Verabreichte Dosen'])
    ax2.set_title('Gelieferte und Verabreichte Dosen')
    ax2.set_ylabel('Anzahl Dosen in Mio')
    ax2.set_xlabel('Hersteller')
    
    
    
    data_supply.loc[data_supply.date=='2020-12-26','date']='2020-12-27'
    width = 0.75  # the width of the bars
    

    
    
    """Tägliche Dosen """
    ax3 = fig1.add_subplot(gs[1, :])
    ax3.set_title("Täglich verabreichte Dosen")
    ax3.plot(data_dates,data.dosen_differenz_zum_vortag/1e3,label='Aktuell tägliche Dosen:            '+str(data.dosen_differenz_zum_vortag[current_day]))
    ax3.plot(data_dates,data.dosen_zweit_differenz_zum_vortag/1e3,'--',c=blue_plot,label='Davon Zweitimpfungen:          '+str(data.dosen_zweit_differenz_zum_vortag[current_day]))
    ax3.step(data_dates,data.mean_weekly/1e3,label='7-Tages Durchschnitt der\n täglich verabreichten Dosen:  '+str(int(data.mean_weekly[current_day])))
    ax3.plot(data_dates,prediction_plot.mean_weekly[0:days].unique()/1e3,label='Ausgleichsfunktion Mittelwert: '+str(int(prediction_plot.mean_weekly[current_day])))
    
    
    # ax3.text((current_day-current_day/12),(data.mean_weekly[current_day]+0.025*data.mean_weekly[current_day])/1e3,round(data.mean_weekly[current_day]/1e3,3),color='orange')
    # ax3.text((current_day-current_day/12),(data.mean_weekly[current_day]-0.1*data.mean_weekly[current_day])/1e3,round(prediction_plot.mean_weekly[current_day]/1e3,3),color='g')
    # ax3.text((current_day-current_day/12),(data.mean_weekly[current_day]+0.1*data.mean_weekly[current_day])/1e3,round(data.dosen_differenz_zum_vortag[current_day]/1e3,3),color='darkblue')
    ax3.legend(labelcolor='linecolor')
    # ax3.plot(df_date.date,(prediction_plot.mean_weekly)/1e3)
    # ax3.plot(df_date.date,(prediction_plot.mean_weekly + prediction_plot.prediction)/1e3)
    # ax3.plot(df_date.date,(prediction_plot.mean_weekly - prediction_plot.prediction)/1e3)
    # ax3.text((current_day-current_day),prediction_plot.mean_weekly[current_day]/1e3,round(model.rsquared_adj,3))
    
    # adj=model.rsquared_adj
    # r_adj='\n'.join(("Adj Bestimmtheitsmaß: %2f"%(adj,),))
    
    # props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    # ax3.text(0.015,0.8,r_adj, transform=ax3.transAxes, fontsize=11,
    #         verticalalignment='top', bbox=props)
    
    ax3.grid(True)
    ax3.set_xlabel('Datum')
    ax3.set_ylabel('Tägliche Impfungen in Tausend')
    
    
    
    """Kummulative Darstellung der verabreichten Dosen """
    ax4 = fig1.add_subplot(gs[2, :])
    
    if days==len(data.personen_erst_kumulativ.unique()):
        ax4.plot(data_dates,data.dosen_kumulativ.unique()/1e6,label='Anzahl verabreichter Dosen gesamt: '+str(round(data.dosen_kumulativ[current_day]/1e6,1)))
        ax4.plot(data_dates,data.dosen_erst_kumulativ.unique()/1e6,label='Anzahl Erstimpfung:                           '+str(round(data.personen_erst_kumulativ[current_day]/1e6,1)))
        ax4.plot(data_dates,data.personen_voll_kumulativ.unique()/1e6,label='Anzahl Zweitimpfung:                        '+str(round(data.personen_voll_kumulativ[current_day]/1e6,1)))
    else:    
        ax4.plot(data.date,data.dosen_kumulativ/1e6,label='Anzahl verabreichter Dosen gesamt')
        ax4.plot(data.date,data.dosen_erst_kumulativ/1e6,label='Anzahl Erstimpfung')
        ax4.plot(data.date,data.personen_voll_kumulativ/1e6,label='Anzahl Zweitimpfung')
    
    # ax4.text((current_day-current_day/12),data.dosen_kumulativ[current_day]/1e6,round(data.dosen_kumulativ[current_day]/1e6,3),ha='left', va='bottom',color='darkblue')
    # ax4.text((current_day-current_day/12),data.personen_erst_kumulativ[current_day]/1e6,round(data.personen_erst_kumulativ[current_day]/1e6,3),horizontalalignment='left', va='bottom',color='orange')
    # ax4.text((current_day-current_day/12),data.personen_voll_kumulativ[current_day]/1e6,round(data.personen_voll_kumulativ[current_day]/1e6,3),ha='left', va='bottom',color='darkgreen')
    #ax4.plot(())
    
    ax4.bar(dates,data_supply.dosen_kummulativ.unique()/1e6,alpha=0.5,label='Anzahl gesamte Lieferungen:            '+str(round(data_supply.dosen_kummulativ[len_supply-1]/1e6,1)))
    # ax4.text((current_day-current_day/12),data_supply.dosen_kummulativ[len_supply-1]/1e6,round(data_supply.dosen_kummulativ[len_supply-1]/1e6,3))
    
    
    ax4.set_title("Durchgeführten Impfungen Kummulativ / Gelieferte Dosen kummulativ")
    ax4.set_xlabel('Datum')
    ax4.set_ylabel('Impfungen in Millionen')
    ax4.legend(labelcolor='linecolor')
    # percent='\n'.join(('Impfung Bevölkerung BRD in %:\n',\
    #                   'Erstimpfung: %.2f '%(data.impf_quote_erst[current_day]*100, ),\
    #                   'Vollständig:   %.2f'%(data.impf_quote_voll[current_day]*100, )))
        
    # props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    # ax4.text(0.015, 0.7, percent, transform=ax4.transAxes, fontsize=11,
    #         verticalalignment='top', bbox=props)
    ax4.grid(True)
    
    
    """Lieferung nach Hersteller """
    ax5 = fig1.add_subplot(gs[3, :])
    ax5.set_title('Lieferungen nach Hersteller in Mio.')
    ax5.plot(data.date,np.zeros(len(data.date)))
    rects1 = ax5.bar(data_supply.date , data_supply.biontech/1e6, width, label='Biontech')
    rects2 = ax5.bar(data_supply.date , data_supply.moderna/1e6,width,bottom=data_supply.biontech/1e6, label='Moderna')
    rects3 = ax5.bar(data_supply.date , data_supply.astra/1e6,width, bottom=data_supply.biontech/1e6+data_supply.moderna/1e6, label='Astrazeneca')
    rects1 = ax5.bar(data_supply.date , data_supply.johnson/1e6, width, label='Johnson')
    ax5.legend(loc=2)
    ax5.set_xlabel('Datum')
    ax5.set_ylabel('Lieferung in Mio.')
    ax5.grid(True)
    
    supply='\n'.join(('Lieferung gesamter Impfstoff in Mio:\n',\
                      'Biontech:       %.2f '%(max(data_supply.biontech_gesamt)/1e6, ),\
                      'Astrazeneca: %.2f'%(max(data_supply.astra_gesamt)/1e6,),\
                      'Moderna:       %.2f '%(max(data_supply.moderna_gesamt)/1e6, ),\
                      'Johnson:         %.2f'%(max(data_supply.johnson_gesamt)/1e6,),))
        
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    ax5.text(0.015, 0.75, supply, transform=ax5.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    
 
    
    
    tick_spacing=7
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax5.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    #fig1.autofmt_xdate(rotation=45)  
    angle=90
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=angle)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=angle)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=angle)
    fig1.tight_layout(pad=3.0)
    plt.savefig('Grafiken/Aktuelle_Impfstatistik.pdf')       
    plt.savefig('C:/Users/aidac/Dropbox/Impfstatistik/Aktuelle_Impfstatistik.pdf')       
    
        