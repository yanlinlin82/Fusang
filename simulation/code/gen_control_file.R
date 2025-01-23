#!/usr/bin/env Rscript

### ARGUMENTS: N taxa N_sims Aln_length

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 10) {
  stop("Usage: Rscript gen_control_file.R seed n_taxa n_sim len_of_msa_lower_bound len_of_msa_upper_bound indel_substitution_rate_lower_bound indel_substitution_rate_upper_bound max_indel_length in.newick.csv out.control.txt")
}
seed = as.numeric(args[1])
n_taxa = as.numeric(args[2])
n_sim = as.numeric(args[3])
len_of_msa_lower_bound = as.numeric(args[4])
len_of_msa_upper_bound = as.numeric(args[5])
indel_substitution_rate_lower_bound = as.numeric(args[6])
indel_substitution_rate_upper_bound = as.numeric(args[7])
max_indel_length = as.numeric(args[8])
in_newick = args[9]
out_control = args[10]

library(MCMCpack)

set.seed(seed)
options(scipen=999) # disable scientific notation

#Model block generating function
model_gen=function(modelset,file,max_indel_length,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound)
{
  modelnames = paste(modelset, 'Model', seq_along(modelset), sep='')
  results = list()

  N = length(modelset)
  I = runif(N, 0, 1)
  A = runif(N, 0, 5)
  Pi = rdirichlet(N, alpha = c(5,5,5,5)) # Nucl proportions DIRICHLET
  indel_rate = runif(N, indel_substitution_rate_lower_bound, indel_substitution_rate_upper_bound)

  for (i in seq_along(modelset)) {
    model = modelset[i]
    model_name = modelnames[i]

    output_lines = paste('[MODEL] ',model_name)

    if (model %in% c('HKY','K80')){
      submodel = paste(c(model, format(runif(1,0,3), digits=2)), collapse=" ")
    } else if (model == 'TrN'){
      submodel = paste(c(model, format(runif(2,0,3), digits=2)), collapse=" ")
    } else if (model %in% c('TIM' ,'TIMef')){
      submodel = paste(c(model, format(runif(3,0,3), digits=2)), collapse=" ")
    } else if (model == 'TVM'){
      submodel = paste(c(model, format(runif(4,0,3), digits=2)), collapse=" ")
    } else if (model %in% c('SYM','GTR')){
      submodel = paste(c(model, format(runif(5,0,3), digits=2)), collapse=" ")
    } else if (model == 'UNREST'){
      submodel = paste(c(model, format(runif(11,0,3), digits=2)), collapse=" ")
    } else {
      submodel = model
    }
    output_lines = c(output_lines, paste(' [submodel]', submodel))

    output_lines = c(output_lines, paste(' [rates]', paste(I[i], A[i], 0), collapse=" "))
    output_lines = c(output_lines, paste(' [indelmodel] POW 1.5', max_indel_length))
    output_lines = c(output_lines, paste(' [indelrate]', indel_rate[i]))

    if (model %in% c('F81','HKY','TrN','TIM','TVM','GTR'))
    {
      output_lines = c(output_lines, paste(' [statefreq]', paste(format(Pi[i,], digits=5), collapse=" ")))
    }

    results[[model_name]] = output_lines
  }
  return(results)
}

indelib_gen = function(n_taxa,n_sim,len_of_msa_lower_bound,len_of_msa_upper_bound,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound,max_indel_length,in_newick,out_control) # n_sim = number of simulations per topology
{
  output_lines = "[TYPE] NUCLEOTIDE 2"
  output_lines = c(output_lines, "[SETTINGS]")
  output_lines = c(output_lines, " [output] FASTA")
  output_lines = c(output_lines, paste(" [randomseed]", round(runif(1,1,100000)), collapse=" "))
  write(output_lines, out_control, sep='\n')

  n_datasets = n_sim

  #Set MODEL block
  modelset = sample(c('JC','TIM','TIMef','GTR','UNREST'),n_datasets,replace=T)
  MODEL_LIST = model_gen(modelset,out_control,max_indel_length,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound)

  for (model_name in names(MODEL_LIST)) {
    write(MODEL_LIST[[model_name]], out_control, append=T, sep='\n')
  }

  #Set TREE block
  ID_TREE = paste(rep("t_sim",times=n_datasets),1:n_datasets,sep="")
  print("Newick")
  NEWICK = read.csv(in_newick,header=TRUE)
  NEWICK = NEWICK[,2]
  print("Done newick")
  write.table(data.frame('[TREE]',ID_TREE,NEWICK),out_control,append=T,quote=F,row.names=F,col.names =F)

  #Set PARTITIONS block
  PNAME = paste("p",1:n_datasets,sep="")
  write.table(data.frame('[PARTITIONS]',PNAME,"[",ID_TREE,names(MODEL_LIST),round(runif(n_sim,len_of_msa_lower_bound,len_of_msa_upper_bound)),"]"),out_control,append=T,quote=F,row.names=F,col.names =F)

  #Set EVOLVE block
  write('[EVOLVE]',out_control,append=T)
  write.table(data.frame(PNAME,1,apply(data.frame(ID_TREE),1,paste,collapse="")),out_control,append=T,quote=F,row.names=F,col.names =F)
}

indelib_gen(n_taxa,n_sim,len_of_msa_lower_bound,len_of_msa_upper_bound,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound,max_indel_length,in_newick,out_control)
