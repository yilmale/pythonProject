# Title     : TODO
# Objective : TODO
# Created by: yilmaz
# Created on: 8/17/20

x <- 1:10
x

testdt <- function(x) {
                library(party)
                png(file = "decision_tree.png")
                output = ctree(nativeSpeaker ~ score, data = x)
                plot(output)
                dev.off()
                output
            }


data(airquality)
airq <- subset(airquality, !is.na(Ozone))
air.ct <- ctree(Ozone ~ ., data = airq, controls = ctree_control(maxsurrogate = 3))
plot(air.ct)
