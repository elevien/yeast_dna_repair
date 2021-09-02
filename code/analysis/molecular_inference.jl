3using Turing, Distributions, DifferentialEquations
using MCMCChains
#using PyPlot
using CSV
using DataFrames
using MCMCChains
using Plots
using StatsPlots


function dna_model(du,u,p,t)
    m, b, r = u
    alpha, beta, rho, tlag   = p
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

# make ODE problem
p0 = [0.003*60, 0.001*60, 0.006*60,5.]
u0 = [1.0,0.0,0.0]
prob1 = ODEProblem(dna_model,u0,(0.0,12.0),p0)

# define turing fit function
@model function mol_fit_func(data,p0)
    # fit model params from microfluidic data

    # priors
    sigma ~ truncated(Normal(0.03,0.02),0.,Inf) # experimental noise
    alpha ~ truncated(Normal(p0[1],1.),0.,Inf) # growth rate
    beta ~ truncated(Normal(p0[2],1.),0,Inf) # break rate
    rho ~ truncated(Normal(p0[3],1.),0,Inf) # repair rate
    tlag ~ truncated(Normal(p0[4],4.),0,12.) # repair rate

    p = [alpha,beta,rho,tlag]
    prob = remake(prob1,u0 = [1000000.0,0.0,0.0], p=p)
    predicted = solve(prob,Tsit5(),saveat=collect(0:2.:12))

    for i = 1:length(predicted)
        data[i,1] ~ Normal(predicted[i][2]/sum(predicted[i]), sigma)
        data[i,2] ~ Normal(predicted[i][3]/sum(predicted[i]), sigma)
    end
end

@model function micro_fit_func(data,chain,times)
    # fit model params from microfluidic data

    # priors
    sigma ~ truncated(Normal(0.03,0.02),0.,Inf) # experimental noise
    alpha = truncated(Normal(p0[1],1.),0.,Inf) # growth rate
    beta = truncated(Normal(p0[2],1.),0,Inf) # break rate
    rho = truncated(Normal(p0[3],1.),0,Inf) # repair rate
    tlag ~ truncated(Normal(p0[4],4.),0,12.) # repair rate

    p = [alpha,beta,rho,tlag]
    prob = remake(prob1,u0 = [1.0,0.0,0.0], p=p)
    predicted = solve(prob,Tsit5(),saveat=times)

    for i = 1:length(predicted)
        data[i,1] ~ Normal(sum(predicted[i]), sigma)
        data[i,2] ~ Normal(predicted[i][3], sigma)
    end
end


function run_bayes_inference(mol_df,micro_df,outfolder)

    # restructure data
    mol_times = mol_df[:,:time]
    mol_gfp = mol_df[:,:gfp]./100.
    mol_dsb = mol_df[:,:dsb]./100.
    mol_odedata = hcat(dsb,gfp);

    micro_bf = micro_df[:,:bf]
    micr_gfp = micro_df[:,:gfp]
    micro_times =  micro_df[:,:time]/60.
    micro_odedata = hcat(bf,gfp);

    # run inference on molecular data
    u0_data = [10e8*(1.0-mol_gfp[1]-mol_dsb[1]),10e8*mol_dsb[1],10e8*mol_gfp[1]]
    fitmodel = fit_func(mol_odedata, remake(prob1,u0=u0_data))
    @time chain = mapreduce(c -> sample(fitmodel, NUTS(.65),1000), chainscat, 1:3)
    write(outfolder*"/mol_chain.jls",chain)

    # run inference on microfluidic data

end




cd(dirname(@__FILE__))
pwd()
nuclease = "SpCas9"
defect = "NR"

mol_df = CSV.read(pwd()*"/../../experimental_data/processed_data/DSB_df.txt", DataFrame);
mol_times = mol_df[(mol_df.defect .== defect).&(mol_df.nuclease .== nuclease),:time]
gfp = mol_df[(mol_df.defect .== defect).&(mol_df.nuclease .== nuclease),:gfp]./100.
dsb = mol_df[(mol_df.defect .== defect).&(mol_df.nuclease .== nuclease),:dsb]./100.
mol_odedata = hcat(dsb,gfp);

micro_df = CSV.read("./../../experimental_data/processed_data/avg_data_one_cell.csv", DataFrame);
experiments = unique(micro_df.experiment)
#df = micro_df[micro_df.experiment .== experiments[1],:].time;
bf = micro_df[(micro_df.defect .== "NR").&(micro_df.nuclease .== "SpCas9"),:bf]
gfp = micro_df[(micro_df.defect .== "NR").&(micro_df.nuclease .== "SpCas9"),:gfp]
micro_times =  micro_df[(micro_df.defect .== "NR").&(micro_df.nuclease .== "SpCas9"),:time]/60.
micro_odedata = hcat(bf,gfp);
