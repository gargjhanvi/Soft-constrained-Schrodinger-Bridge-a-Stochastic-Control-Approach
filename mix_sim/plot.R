args = commandArgs(TRUE)
pre = args[1]
n = 10
ts = seq(0.1, 1, by = 0.1)

png(filename=paste(pre,'.png',sep=''), res=300, units='in', width=12, height=2)
par(mfrow=c(1, 6))
par(mar=c(3, 3, 1, 1))
par(las=1)
par(mgp=c(2,0.5,0))

ks = c(2, 4, 6, 8, 9, 10)
for (k in ks){
    f = paste(pre, '_', k, '.txt', sep='')
    d = as.matrix(read.table(f))
    r = which(apply(abs(d), 1, sum)>10)
    if (length(r) > 0){
        print(r)
        d = d[-r, ]
    }
    plot(d, xlab=bquote(t==.(ts[k])), ylab='', cex=0.2)    
}
dev.off()


