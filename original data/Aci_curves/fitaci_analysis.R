###### package loading
#install.packages("readxl")
#install.packages("ggplot2")
#install.packages("plantecophys")
#install.packages("plyr")
#install.packages("tm")
#install.packages('ggpubr')
#install.packages('Rtools')

library(plantecophys)
setwd('C:/Users/jans008/OneDrive - Wageningen University & Research/PhD/experiments/Oefenen DUAL-PAM onder SUN_FR/final measurements/Aci_curves')

myArgs <- commandArgs(trailingOnly = TRUE)

data = myArgs[1:length(myArgs)-1]
plantID = myArgs[length(myArgs)]
values = as.numeric(data)

x = length(values)
plant <- data.frame(Photo = values[1:(x/2)],  Ci = values[(x/2+1):x] )
write.csv(plant, file='C:/Users/jans008/OneDrive - Wageningen University & Research/PhD/experiments/Oefenen DUAL-PAM onder SUN_FR/final measurements/Aci_curves/data.csv')
f = fitaci(plant, gmeso = 0.3, fitmethod='bilinear')

png(paste(plantID, "Aci.png"), width = 800, height = 600)
plot(f)
dev.off()


cat(unname(coef(f)))





