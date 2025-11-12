include("../analytic_ttvs/ttv_ss/src/bounds.jl")
using MCMCChains, Statistics, TTVFaster, DelimitedFiles, JLD2, LaTeXStrings, MCMCDiagnosticTools
function calc_lprior(param::Vector{Float64})
    nplanet = Int(floor(length(param)/5))
  # We will place a joint prior on eccentricity vector
  # such that each planet has an eccentricity which lies between
  # 0 and emax2,with a gradual decrease from emax1 to emax2:
  # Eccentricity priors:
    emax1 = 0.2; emax2 = 0.3
    # Define -log(prior):
    lprior = 0.0
    # Loop over planets:
    for iplanet=1:nplanet
    # Place prior on eccentricities:
      ecc = sqrt(param[(iplanet-1)*5+4]^2+param[(iplanet-1)*5+5]^2)
      lprior_tmp,dpdx = log_bounds_upper(ecc,emax1,emax2)
      lprior += lprior_tmp
      lprior += -log(ecc) 
    end
    for iplanet=1:nplanet-1
    # The periods of the planets should be ordered from least to greatest:
        if param[(iplanet-1)*5+2] > param[iplanet*5+2]
          lprior += -Inf
        end
    end
    for iplanet=1:nplanet
    # The masses should be positive:
      if param[(iplanet-1)*5+1] < 0 #unsure if this is correct implement
        lprior += -Inf
      end
    end
        # The systematic uncertainty should be positive:
      sigsys = param[end]
      if sigsys < 0 
        lprior += -Inf
      end
    return lprior
end

function MCMC(foutput::String,param::Array{Float64,1},lprob_best::Float64,
        nsteps::Int64,nwalkers::Int64,
        nplanet::Int64,ntrans::Array{Int64,1},
        tt0::Array{Float64,1},tt::Array{Float64,1},sigtt::Array{Float64,1},
        use_callisto_data::Bool=true)
    nparam = length(param)
    println(nparam," parameters from fit: ",param)
    println("Maximum log Prob from fit: ",lprob_best)
    println("No Walkers: ",nwalkers," No. steps: ",nsteps)
    jmax = 5
    pname =["μ_1", "P_1", "t0_1", "ecosω_1", "esinω_1", 
            "μ_2", "P_2", "t0_2", "ecosω_2", "esinω_2", 
            "μ_3", "P_3", "t0_3", "ecosω_3", "esinω_3", 
            "μ_4", "P_4", "t0_4", "ecosω_4", "esinω_4"]
    if use_callisto_data
        errors = [
        1e-7,1e-5,1e-5,1e-2,1e-2,
        1e-7,1e-5,1e-5,1e-2,1e-2,
        1e-7,1e-5,1e-5,1e-2,1e-2,
        1e-7,1e-5,1e-5,1e-2,1e-2            
        ]
    else
        errors = [
        1e-7,1e-5,1e-5,1e-2,1e-2,
        1e-7,1e-5,1e-5,1e-2,1e-2,
        1e-7,1e-5,1e-5,1e-2,1e-2]
    end
    pname=[pname;"σ_sys"]
    par_trial = [param[1:end];1e-8]
    nsize = nparam+1
    # Number of walkers should be at least 3 times number of parameters:
    @assert (nwalkers >= 3*nparam)  
    # Set up arrays to hold the results:
    par_mcmc = zeros(nwalkers,nsteps,nsize)
    lprob_mcmc = zeros(nwalkers,nsteps)
    thinning=1000
    thin_chain = zeros(nwalkers,div(nsteps,thinning), nsize)
    thin_lprob = zeros(nwalkers,div(nsteps,thinning))
    
    Nobs = sum(ntrans) - 2*(nplanet - 2)
    for j=1:nwalkers
        # Select from within uncertainties:
        lprob_trial = -1e100
        # Only initiate models with reasonable Log Prob values:
        while lprob_trial < lprob_best - 1000 #since logProb, maybe 1000 is too large
            par_trial[1:nparam] .= param .+ errors .* randn(nparam) 
            par_trial[nparam+1] = 1e-8 .* abs(randn())
            lprob_trial = calc_lprior(par_trial)
                # println("Calculated Log Prior: ",lprob_trial)
            if lprob_trial > -Inf
                model = new_wrapper(tt0,nplanet,ntrans,par_trial[1:nparam],jmax)
            # println(length(tt)," ",length(model))
            # ll = (-0.5 * sum((tt-model).^2 ./(sigtt.^2 .+ sigsys.^2) .+ log.(sigtt.^2 .+ sigsys.^2)))
                 lprob_trial += (-0.5 * sum((tt-model).^2 ./(sigtt.^2 .+ par_trial[end].^2) .+ log.(sigtt.^2 .+ par_trial[end].^2)))
            end
          # println("Trial Log Prob: ",lprob_trial)
        end #while
        lprob_mcmc[j,1]=lprob_trial
        par_mcmc[j,1,:]=par_trial
    # println("Success: ",par_trial,lprob_trial)
    end #walkers
    # Initialize scale length & acceptance counter:
    ascale = 2.0
    accept = 0
    # Next,loop over steps in markov chain:
    for i=2:nsteps
        for j=1:nwalkers
            ipartner = j
            # Choose another walker to 'breed' a trial step with:
            while ipartner == j
                ipartner = ceil(Int,rand()*nwalkers)
            end
            # Now,choose a new set of parameters for the trial step:
            z=(rand()*(sqrt(ascale)-1.0/sqrt(ascale))+1.0/sqrt(ascale))^2 # draws from 1=sqrt(z....)
            par_trial=vec(z*par_mcmc[j,i-1,:]+(1.0-z)*par_mcmc[ipartner,i-1,:])
            # Compute model & Log Prob for trial step:  
            lprob_trial = calc_lprior(par_trial)  
            if lprob_trial > -Inf
                model_trial = new_wrapper(tt0,nplanet,ntrans,par_trial[1:nparam],jmax)
                lprob_trial += (-0.5 * sum((tt-model_trial).^2 ./(sigtt.^2 .+ par_trial[end].^2) .+ log.(sigtt.^2 .+ par_trial[end].^2)))
            end # while 

    # Next,determine whether to accept this trial step:
    # Calculate ratio between Log Prob of trial step and previous step:
    alp = z^(nsize-1)*exp((lprob_trial - lprob_mcmc[j,i-1]))
                        #=
    #logratio= log(z)*(nsize-1)+lprob_trial - lprob_mcmc[j,i-1]
    #if log(rand()) < logratio # if rejected, last values equal new values
    #  par_mcmc[j,i,:] = par_trial
    #  lprob_mcmc[j,i] = lprob_trial
    #end
    # if rand() < 0.0001 
    #   println("Step: ",i," Walker: ",j," Trial Log Prob: " ,lprob_trial," Prob: ",alp," Frac: ",accept/(mod(i-1,1000)*nwalkers+j))
    # end
                        =#
        if alp >= rand()
            # If step is accepted,add it to the chains!
            par_mcmc[j,i,:] = par_trial
            lprob_mcmc[j,i] = lprob_trial
            accept = accept + 1
        else
            # If step is rejected,then copy last step:
            par_mcmc[j,i,:] = par_mcmc[j,i-1,:]
            lprob_mcmc[j,i] = lprob_mcmc[j,i-1]
        end
    end # step loop
    if mod(i,thinning) == 0
        println("Number of steps: ",i," Acceptance Rate: ",accept/(thinning*nwalkers))
    end
    end #nwalker 
      # Now,determine time of burn-in by calculating first time median is crossed:
    iburn = 0
    for i=1:nsize
        med_param=median(par_mcmc[1:nwalkers,1:nsteps,i])
        for j=1:nwalkers
            istep=2
            while (par_mcmc[j,istep,i] > med_param) == (par_mcmc[j,istep-1,i] > med_param) && (istep < nsteps)
                istep=istep+1
            end
            if istep >= iburn
                iburn = istep
            end
        end #walkers
    end#nsize
    println("Burn-in Number (ends): ",iburn)
    
    # Convert to Chains object, per MCMCChains
    iters, samples, nparams = size(par_mcmc)
    tmp = zeros(samples,nparams,iters)
    for i=1:nparams
        for j=1:iters
          for k=iburn:samples
            tmp[k,i,j] = par_mcmc[j,k,i] 
          end
        end
    end
    chn = Chains(tmp[iburn:end,:,:],pname)
    mcmcfile = string(foutput)
    @save mcmcfile par_mcmc lprob_mcmc param nwalkers nsteps iburn chn pname #indepsamples pname
end
