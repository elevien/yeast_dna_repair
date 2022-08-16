

## Plotting

function posterior_corner(chain_df,size)
    cols = length(names(chain_df))
    fig, axs= plt.subplots(figsize=size,ncols=cols,nrows=cols,sharex="col",sharey="row")
    for j in 1:cols
        ax = axs[j,j]
        U = kde(chain_df[:,j])
        ax.plot(U.x,U.density)
        ax.fill_between(U.x,zeros(length(U.x)),U.density,alpha=0.3)
        ax.spines["right"].set_visible(false)
        ax.spines["top"].set_visible(false)
        for k in 1:(j-1)
            ax.get_shared_y_axes().remove(axs[j,k])
        end
        ax.autoscale()
        axs[j,1].set_ylabel(names(chain_df)[j])
        axs[cols,j].set_xlabel(names(chain_df)[j])
        for i in j+1:cols
            ax = axs[i,j]
            ax.spines["right"].set_visible(false)
            ax.spines["top"].set_visible(false)
            #U = kde(hcat(chain_df[:,j],chain_df[:,i]),bandwidth=(.001,.001))
            #ax.contour(U.x,U.y,U.density,levels=0.1:max(U.density...)/5:max(U.density...))
            ax.plot(chain_df[:,j],chain_df[:,i],".",alpha=0.1)


            ax = axs[j,i]
            ax.axis("off")

        end
    end
    return axs

end



function mbr_model(du,u,θ,t)
    m, b, r = u
    sigma, alpha, beta, rho, tlag   = θ
    if t>tlag
        du[1] = (alpha-beta)*m
        du[2] = beta*m - rho*b
        du[3] = alpha*r + rho*b
    else
        du[1] = 0.0
        du[2] = 0.0
        du[3] = 0.0
    end

end

# model with repaired but non-gfp cells
function mbrg_model(du,u,θ,t)
    m, b, r, g = u
    sigma, alpha, beta, gamma, rho, tlag   = θ
    if t>tlag
        du[1] = (alpha-beta)*m
        du[2] = beta*m - rho*b
        du[3] = alpha*r + rho*b  - gamma*r
        du[4] = alpha*g + gamma*r
    else
        du[1] = 0.0
        du[2] = 0.0
        du[3] = 0.0
        du[4] = 0.0
    end

end

# set priors
α_prior_mu = log(2)/3
α_prior_CV = 1
β_prior_mu = log(2)/4
β_prior_CV = 1
ρ_prior_mu = log(2)/4
ρ_prior_CV = 1
σ_prior_mu = 0.05
σ_prior_CV = 1
τ_prior_mu = 4
τ_prior_CV = 0.1
σ_prior = Gamma(1/σ_prior_CV^2,σ_prior_mu*σ_prior_CV^2) # experimental noise
α_prior = Gamma(1/α_prior_CV^2,α_prior_mu*α_prior_CV^2) # growth rate
β_prior = Gamma(1/β_prior_CV^2,β_prior_mu*β_prior_CV^2) # break rate
ρ_prior = Gamma(1/ρ_prior_CV^2,ρ_prior_mu*ρ_prior_CV^2) # repair rate
τ_prior = Gamma(1/τ_prior_CV^2,τ_prior_mu*τ_prior_CV^2)
priors = [σ_prior,α_prior,β_prior,ρ_prior,τ_prior];


@model function molecular_model(data,prob)
    # fit model params from microfluidic data

    # priors
    σ ~ σ_prior # experimental noise
    α ~ α_prior # growth rate
    β ~ β_prior # break rate
    ρ ~ ρ_prior # repair rate
    τ ~ τ_prior # repair rate

    p = [σ,α,β,ρ,τ]
    prob = remake(prob,p=p)
    predicted = solve(prob,Tsit5(),saveat=collect(0:2.:12))

    for i = 1:length(predicted)
        u1 = predicted[i][2]/sum(predicted[i])
        u2 = predicted[i][3]/sum(predicted[i])
        data[i,1] ~ Normal(u1, σ)
        data[i,2] ~ Normal(u2, σ)
    end
end


function get_posterior(mol_df)
    # get data ready
    gfp = mol_df.gfp
    dsb = mol_df.dsb
    u0_data = [10e8*(1.0-gfp[1]-dsb[1]),10e8*dsb[1],10e8*gfp[1]]
    mol_times = mol_df.time
    mol_odedata = hcat(dsb,gfp);


    # run inference on molecular data

    fitmodel = molecular_model(mol_odedata,ODEProblem(mbr_model,u0_data,(0.0,12.0),θ))
    @time chain = mapreduce(c -> sample(fitmodel, NUTS(.65),500), chainscat, 1:3)
    return chain

end


function inference_pipeline(mol_df,micro_df,outfile)

    """
    This function takes as inpus a nuclease, defect and output file and runs the bayesian inference and posterior prediction pipline.
    It is assumed that the data is already loaded. In particular, we do the following:
    1. run Bayesian inference of the parameters in the ODE model and save the posterior samples.
    2. Run posterior predictive simulations to compare to raw DSB data
    3. Run posterior predictive simulations to compare to microfluidic data
    """


    # get data ready
    chain = get_posterior(mol_df)


    #posterior_corner(DataFrame(chain)[:,[:σ,:α,:β,:ρ,:τ]],(7,7))

    # ------------------------------------------------------------------------------------------
    # save dataframe with chain information
    chain_df = DataFrame(chain)
    CSV.write(outfile*"_chain.csv",chain_df)

    gfp = mol_df.gfp
    dsb = mol_df.dsb
    u0_data = [10e8*(1.0-gfp[1]-dsb[1]),10e8*dsb[1],10e8*gfp[1]]
    mol_times = mol_df.time
    mol_odedata = hcat(dsb,gfp);
    gfp= micro_df.gfp
    bf = micro_df.bf
    micro_times = micro_df.time
    micro_odedata = hcat(bf,gfp);

    # ------------------------------------------------------------------------------------------
    # plot posterior distribution of alpha,beta,rho,lag ...
    fig,axs = PyPlot.plt.subplots(nrows = 3,figsize=(5,4))
    PyPlot.plt.tight_layout()
    ax = axs[1]
    U = kde(vcat(Array(chain[:β])...))
    ax.plot(U.x,U.density,label=L"\beta")
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    ax.set_xlabel("1/hours")


    U = kde(vcat(Array(chain[:ρ])...))
    ax.plot(U.x,U.density,label=L"\rho")

    prior = rand(ρ_prior,10^5)
    U = kde(prior)
    ax.plot(U.x,U.density,"k--",label="prior")
    ax.set_xlim([0,3])
    ax.legend(frameon=false)


    ax = axs[2]
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    U = kde(vcat(Array(chain[:α])...))
    ax.plot(U.x,U.density,label=L"\alpha")


    prior = rand(α_prior,10^5)
    U = kde(prior)
    ax.plot(U.x,U.density,"k--",label="prior")

    ax.legend(frameon=false)
    ax.set_xlabel("1/hours")


    ax = axs[3]
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    U = kde(vcat(Array(chain[:τ])...))
    ax.plot(U.x,U.density,label=L"\tau_{\rm lag}")

    prior = rand(τ_prior,10^5)
    U = kde(prior)
    ax.plot(U.x,U.density,"k--",label="prior")

    ax.legend(frameon=false)
    ax.set_xlabel("hours")

    PyPlot.plt.savefig(outfile*"_posterior_params.svg",bbox_inches="tight")

    # ------------------------------------------------------------------------------------------
    # plot posterior predictive distribution compared to molecular data

    fig,axs = PyPlot.plt.subplots(ncols=2,figsize=(10,3))
    ax = axs[1]
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    ax.plot(mol_times,mol_odedata[:,1],"C0o",label = "fraction broken",fillstyle="none")
    chain_df = DataFrame(chain)[:,[:σ,:α,:β,:ρ,:τ]]
    for k in 1:30
        p = chain_df[rand(1:1500), :]
        prob_rep = ODEProblem(mbr_model,u0_data,(0.0,12.0),p)
        sol = solve(prob_rep,Tsit5(),saveat=0.1)
        gfp_rep = [sol.u[i][3] for i in 1:length(sol.u)]
        dsb_rep = [sol.u[i][2] for i in 1:length(sol.u)]
        bf_rep = [sum(sol.u[i]) for i in 1:length(sol.u)]
        ax.plot(sol.t,dsb_rep ./bf_rep,"C0-",alpha=0.2)

        # save ode simulations
        CSV.write(outfile*"_dsb_ode_sol_$k.csv",DataFrame([gfp_rep,bf_rep,dsb_rep],:auto))

    end
    ax.set_xlabel("hours")
    ax.legend(frameon=false)


    ax = axs[2]
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    ax.plot(mol_times,mol_odedata[:,2],"C1o",label = "fraction repaired",fillstyle="none")
    chain_df = DataFrame(chain)[:,[:σ,:α,:β,:ρ,:τ]]
    for k in 1:30
        p = chain_df[rand(1:1500),:]
        prob_rep = ODEProblem(mbr_model,u0_data,(0.0,12.0),p)
        sol = solve(prob_rep,Tsit5(),saveat=0.1)
        gfp_rep = [sol.u[i][3] for i in 1:length(sol.u)]
        dsb_rep = [sol.u[i][2] for i in 1:length(sol.u)]
        bf_rep = [sum(sol.u[i]) for i in 1:length(sol.u)]
        ax.plot(sol.t,gfp_rep ./bf_rep,"C1-",alpha=0.2)
    end
    ax.set_xlabel("hours")
    ax.legend(frameon=false)
    PyPlot.plt.savefig(outfile*"_posterior_predict_w_molecular.svg",bbox_inches="tight")

    # ------------------------------------------------------------------------------------------
    # plot posterior predictive distribution compared to bulk microfluidic data
    fig,ax = PyPlot.plt.subplots(figsize=(7,3))

    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    chain_df = DataFrame(chain)[:,[:σ,:α,:β,:ρ,:τ]]
    lags = []
    frac = gfp ./bf
    micro_times = micro_times/60
    for k in 1:20
        p = chain_df[rand(1:1500),:]
        prob_rep = ODEProblem(mbr_model,[micro_odedata[1,1],0.0,micro_odedata[1,2]],(0.0,24.0),p)
        sol = solve(prob_rep,Tsit5(),saveat = micro_times)
        gfp_rep = [sol.u[i][3] for i in 1:length(sol.u)]
        dsb_rep = [sol.u[i][2] for i in 1:length(sol.u)]
        bf_rep = [sum(sol.u[i]) for i in 1:length(sol.u)]
        frac_rep = (gfp_rep ./ bf_rep)
        ax.plot(sol.t,frac_rep,"C0-",alpha=0.2)

        # save ode simulations
        CSV.write(outfile*"_ode_sol_$k.csv",DataFrame([gfp_rep,bf_rep,dsb_rep],:auto))

        lag_ind = argmin([mean((frac_rep[1:end-i+1] .-frac[i:end]).^2) for i in 1:length(frac)-1])
        push!(lags,micro_times[lag_ind])
    end
    CSV.write(outfile*"_lags.csv",DataFrame([lags],:auto))
    ax.plot(micro_times .- mean(lags),frac,"C0o",label="fraction GFP",fillstyle="none")
    ax.set_xlabel("hours")
    ax.legend(frameon=false)
    PyPlot.plt.savefig(outfile*"_posterior_predict_w_microfluidic.svg",bbox_inches="tight")


end
