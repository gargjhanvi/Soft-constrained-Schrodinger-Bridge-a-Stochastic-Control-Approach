logsumexp1 <- function(x) {
    c <- max(x) 
    return(c + log(abs(sum(exp(x - c)))))
}

logsumexp <- function(x, sign) {
    pos <- which(sign == 1)
    neg <- which(sign == -1)
    if (length(pos) == 0){ 
        return(list("log_sum"=logsumexp1(x[neg]), "sign"=-1))  
    }
    if (length(neg) == 0){
        return(list("log_sum"=logsumexp1(x[pos]), "sign"= 1))      
    }
    sp <- logsumexp1(x[pos])
    sn <- logsumexp1(x[neg])
    result_sign <- 0
    result <- -Inf
    if (sp > sn){
        result_sign <- 1
        result <- sp + log(1 - exp(sn - sp))
    }
    if (sp < sn){
        result_sign <- -1
        result <- sn + log(1 - exp(sp - sn))
    }
    return(list("log_sum"=result, "sign"=result_sign))
}


simulate_diffusion <- function(n_steps, prop, dprop, n_mc = 1000, b = 1) {
    d <- 1/n_steps
    time_points <- seq(0, 1, length.out = n_steps + 1)
    values <- numeric(length = n_steps + 1)
    values[1] <- 0
    
    for (i in 2:(n_steps + 1)) {
        t <- time_points[i - 1]
        cauchy_samples_num <- propose(n_mc)
        cauchy_samples_deno <- propose(n_mc)
        x_num = values[i-1] + sqrt(1 - t)*cauchy_samples_num
        x_deno = values[i-1] + sqrt(1 - t)*cauchy_samples_deno
        temp = (x_num^2/2 - log(1 + x_num^2))*b - log(1 + x_num^2)  +  log(abs(x_num**3 - x_num)) + dnorm(cauchy_samples_num, log = TRUE) - dprop(cauchy_samples_num)
        num = logsumexp(temp, sign(x_num**3 - x_num))  
        temp2 = (x_deno^2/2 - log (1+x_deno^2))*b + dnorm(cauchy_samples_deno, log = T) - dprop(cauchy_samples_deno) 
        denom = logsumexp1(temp2)
        u = b * exp(num$log_sum - denom) * num$sign
        values[i] <- values[i - 1] + u * d + sqrt(d) * rnorm(1)
        if (!is.finite(values[i])){
            return(c(NA, i, values[i-1], num$log_sum, denom))
        }
    }
    return(c(values[length(values)], 0, 0, 0, 0) )
}


args = commandArgs(TRUE)
nstep = as.numeric(args[1]) # integer; no. of steps used in time discretization
nmc = as.numeric(args[2]) # integer, no. of Monte Carlo samples for approximating integrals 
b = as.numeric(args[3]) # b must be in 0 to 1.  b = beta/(1 + beta)
log = args[4] # output file
nrep = 10000
out = matrix(0, nrow=nrep, ncol=5)

#propose <- function(n){rcauchy(n)}
#dprop <- function(x){dcauchy(x, log=TRUE)}
propose <- function(n){rt(n, df=2)}
dprop <- function(x){dt(x, df=2, log=TRUE)}

if (b < 1){
    propose <- function(n){
        rnorm(n, sd=sqrt(1/(1-b)))
    }
    dprop <- function(x){
        dnorm(x, sd=sqrt(1/(1-b)), log=TRUE)
    }
}

cat("\n")
for (rep in 1:nrep){
    set.seed(rep)
    out[rep, ] = simulate_diffusion(nstep, propose, dprop, nmc, b)
    cat(rep, "\r")
}
cat("\n")
cat("Fail", sum(is.na(out[,1])),"\n")
write.table(out, file=log, quote=FALSE, row.names = F, col.names = F)

xs = na.omit(out[, 1])
summary(xs)


