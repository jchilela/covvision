# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:19:50 2020

@author: Equipe Instituto Respira Brasil

Baixar Arquivos do Servidos Hostgator para Processsamento , Predição e Explicação dos exames de Raio-X Covid-19
"""

import mysql.connector



import os

import  ftplib

mydb = mysql.connector.connect(
host="youintelligent.com.br",
user="youint10_milloca",
passwd="milloca72SF$",           
database="youint10_wp27")

cursor = mydb.cursor()

sql = "select enderecoimagem from exames where situacao_atual = 'enviado' order by id asc limit 1 "
cursor.execute(sql)

myresult = cursor.fetchall()

if cursor.rowcount > 0:
    
    for x in myresult:
      enderecoimagem = x[0];  
      print(x)
  
      mycursor = mydb.cursor()
      
      sql = "update exames set situacao_atual = %s  where enderecoimagem = %s"
    
      val = ("processando", enderecoimagem)
      try:
          mycursor.execute(sql, val )
        
      except (MySQLdb.Error, MySQLdb.Warning) as e:
                print(e)    


      mydb.commit()

      print(mycursor.rowcount, "record(s) affected") 
           
      imagem =  enderecoimagem.replace("/uploads/", "")

      os.chdir('/home/charles/workspace/Covid/Covid/src/data/data/processed/test')
    

      #excluir arquivos
      
      import glob
                          
      files = glob.glob('/home/charles/workspace/Covid/Covid/src/data/data/processed/test/*')
      for f in files:
          os.remove(f)
      
    #FILENAME = 'WEnvmYq4ay-Exame-pneumonia.jpeg'    
    
      with ftplib.FTP("108.179.253.54", "irb@institutorespirabrasil.org", "Groselha2020$") as ftp:
        ftp.cwd('/irb/uploads')
        with open(imagem, 'wb') as f:
            ftp.retrbinary('RETR ' + imagem, f.write)

    #Begin Predição e Explaination        
      
      os.chdir('/home/charles/workspace/Covid/Covid/src/') 
      import predict
      
      print("iniciando o processsamento em Predict")
      
      #os.chdir('/home/charles/workspace/Covid/Covid/') 
          
      predict.predict_and_explain_set(preds_dir=None, save_results=True, give_explanations=True)   

else:
    print('nada a processar')