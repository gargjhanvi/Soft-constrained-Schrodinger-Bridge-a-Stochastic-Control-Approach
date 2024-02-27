logsumexp1 <- function(x) {
    c <- max(x) 
    return(c + log(abs(sum(exp(x - c)))))
}

mu = list(2)
tau = list(2)
w = list(2)
mu[[1]] = matrix(c(1,1,-1,1,1,-1,-1,-1), byrow=T, nrow=4)
tau[[1]] = 0.05
w[[1]] = c(0.1, 0.2, 0.3, 0.4)
mu[[2]] = matrix(c(1.2, 0.8, -1.5, -0.5), byrow=T, nrow=2)
tau[[2]] = 0.5
w[[2]] = c(0.5, 0.5)
sigma = 1
 
log_f = function(x, id){
    m = length(w[[id]])
    v = numeric(m)
    for (i in 1:m){
        v[i] = sum(dnorm(x, mean=mu[[id]][i, ], sd=tau[[id]], log=TRUE))+log(w[[id]][i])
    } 
    log(sum(exp(v - max(v)))) + max(v)
}

score_f = function(x, id){ 
    m = length(w[[id]]) 
    g = log_f(x, id)
    s = rep(0, length(x))
    for (i in 1:m){
        p = exp(sum(dnorm(x, mean=mu[[id]][i, ], sd=tau[[id]], log=TRUE))-g)
        s = s - (x - mu[[id]][i, ]) / (tau[[id]]^2) * w[[id]][i] * p 
    } 
    return(s)
}

log_f_geom = function(x, a){
    (1-a) * log_f(x, 1) + a * log_f(x, 2)
}

score_f_geom = function(x, a){ 
    (1-a) * score_f(x, 1) + a * score_f(x, 2)
}

log_phi = function(x){
    sum(dnorm(x, mean=0, sd=sigma, log=TRUE))
}

score_phi = function(x){
    - x / (sigma^2)
}


args = commandArgs(TRUE)
beta = as.numeric(args[1])
pre = args[2] #output file
nsim = 1000
nstep = 200 
nmc = 200
ncheck = 10
path = list(ncheck)
intv = nstep/ncheck
for (i in 1:ncheck){
    path[[i]] = matrix(0, nrow=nsim, ncol=2)
}

a = beta/(1+beta)
if (beta == Inf){a = 1}

cat(a, "\n")
for (i in 1:nsim){
    set.seed(i)
    d = 1/nstep
    time = seq(0, 1, length.out=nstep + 1)
    x = c(0, 0)
    for (k in 1:nstep){
        t = time[k]
        y = sweep(matrix(rnorm(2*nmc), ncol=2) * sqrt(1-t), 2, x, '+') 
        log_f_y = apply(y, 1, log_f_geom, a=a)
        log_phi_y = apply(y, 1, log_phi)
        log_r = log_f_y - log_phi_y
        log_deno = logsumexp1(log_r)  - log(nmc)
        score_f_y = apply(y, 1, score_f_geom, a=a) 
        score_phi_y = apply(y, 1, score_phi)  
        b = numeric(2)
        rd = exp(log_r - log_deno) 
        score_y = score_f_y - score_phi_y
        b[1] = mean(rd * score_y[1,])
        b[2] = mean(rd * score_y[2,])
        x = x + b * d + sqrt(d) * sigma * rnorm(2)
        if (k %% intv == 0){
            path[[k/intv]][i, ] = x
        }
        if (any(is.nan(x))){
            cat(i, k, x, b, deno,   "\n")
            break
        }
    }
    cat(i, x, "\n")
}
cat("\n")

for (k in 1:ncheck){
    write.table(path[[k]], file=paste(pre, '_', k, '.txt', sep=''), quote=F, row.names=F, col.names=F)
}


