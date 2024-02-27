args = commandArgs(TRUE)
b = as.numeric(args[1])
out = args[2]
#b = 0.99
#out = 's1_1000'
#b = 1
#out = 'sb_1000'
file = paste(out, '.txt', sep='')

trans <- function(x){
    sign(x) * log(1 + 10*abs(x))    
}
d = as.matrix(read.table(file, header=FALSE))
xs = as.numeric(na.omit(d[,1]))
nx = length(xs)
qs = seq(1e-4, 0.9999, by=1e-4)
ys = qcauchy(qs)

x = trans(xs)
y = trans(ys)

png(filename=paste(out, '.png', sep=''), width=3.5, height=3.5, res=300, units='in')
par(mar=c(3,3,2,1))
par(mgp=c(2,0.5,0))
#qqplot(x, y, xlim=c(min(y), max(y)), xaxt='n', yaxt='n', xlab='SSB samples', ylab='Cauchy distribution', cex=0.5)
qqplot(x, y, xlim=c(-8, 8), ylim=c(-8, 8), main=expression(beta==infinity),  xaxt='n', yaxt='n', xlab='SB samples', ylab='Cauchy distribution', cex=0.5)
zs = c(-1000, -100, -10,  -1, 0, 1,   10, 100, 1000)
par(las=1)
axis(1, at=trans(zs), labels = expression(-10^3, -10^2, -10, -1, 0, 1, 10, 10^2, 10^3))
axis(2, at=trans(zs), labels = expression(-10^3, -10^2, -10, -1, 0, 1, 10, 10^2, 10^3))
abline(a=0, b=1, col='blue')
dev.off()

f <- function(x){
    dnorm(x)^(1-b) * dcauchy(x)^b
}

res = integrate(f, lower=-Inf, upper=Inf)
C = res$value
eps = 5e-4
us = seq(-40, 40, by = eps)
s = 0
qu = numeric(length(qs))
now = 1
for (i in 1:length(us)){
    u = us[i]
    s = s + f(u-eps/2)*eps/C
    while (s > qs[now]){
        qu[now] = u-eps/2
        now = now + 1
        if (now > length(qs)){
            break
        }     
    }
    if (now > length(qs)){
        break
    }     
}
q = trans(qu) 

png(filename=paste(out, '.png', sep=''), width=3.5, height=3.5, res=300, units='in')
par(mar=c(3,3,2,1))
par(mgp=c(2,0.5,0))
qqplot(x, q, xlim=c(min(q), max(q)), xaxt='n', yaxt='n', main=expression(beta==100), xlab='SSB samples', ylab='Geometric mixture', cex=0.5)
zs = c(-1000, -100, -10,  -1, 0, 1,   10, 100, 1000)
par(las=1)
axis(1, at=trans(zs), labels = expression(-10^3, -10^2, -10, -1, 0, 1, 10, 10^2, 10^3))
axis(2, at=trans(zs), labels = expression(-10^3, -10^2, -10, -1, 0, 1, 10, 10^2, 10^3))
abline(a=0, b=1, col='blue')
dev.off()

