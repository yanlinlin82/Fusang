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

set.seed(seed)

library('phangorn')
library('MCMCpack')
library('dplyr')
library('scales')
options(scipen=999) # disable scientific notation

#Model block generating function
model_gen=function(modelset,file,max_indel_length,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound)
{
  modelnames = paste(modelset, 'Model', seq_along(modelset), sep='')

  len=max_indel_length
  for (i in seq_along(modelset)) {
    model = modelset[i]
    model_orig=model
    model_name = modelnames[i]

    #Invariant sites Unif
    I=runif(1,0,1)
    A=runif(1,0,5)
    #Nucl proportions DIRICHLET 
    options(digits=5) # round to 5 decimal places
    Pi=format(rdirichlet(1, alpha=c(5,5,5,5)))
    #IndelRate = format(runif(1,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound))

    output_lines = paste('[MODEL] ',model_name)

    options(digits=2) # round to 2 decimal places
    if (model %in% c('HKY','K80')){
      model=paste(c(model,' ',format(runif(1,0,3))),sep = '')
    } else if (model == 'TrN'){
      model=paste(c(model,' ',format(runif(2,0,3))),sep = '')
    } else if (model %in% c('TIM' ,'TIMef')){
      model=paste(c(model,' ',format(runif(3,0,3))),sep = '')
    } else if (model == 'TVM'){
      model=paste(c(model,' ',format(runif(4,0,3))),sep = '')
    } else if (model %in% c('SYM','GTR')){
      model=paste(c(model,' ',format(runif(5,0,3))),sep = '')
    } else if (model == 'UNREST'){
      model=paste(c(model,' ',format(runif(11,0,3))),sep = '')
    } else {
      model=model
    }

    output_lines = c(output_lines, paste(' [submodel] ',paste(model,collapse=' ')))
    output_lines = c(output_lines, paste(' [rates] ',I,' ',A,' 0'))
    output_lines = c(output_lines, paste(' [indelmodel] POW 1.5 ', paste(len,collapse=' ')))
    output_lines = c(output_lines, paste(' [indelrate] ', paste(runif(1,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound),collapse='')))
    if (model_orig %in% c('F81','HKY','TrN','TIM','TVM','GTR'))
    {
      output_lines = c(output_lines, paste(' [statefreq]',paste(Pi,collapse=' ')))
    }

    write(output_lines, file, append=T, sep='\n')
  }
  return(modelnames)
}

indelib_gen=function(n_taxa,n_sim,len_of_msa_lower_bound,len_of_msa_upper_bound,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound,max_indel_length,in_newick,out_control) # n_sim = number of simulations per topology
{
  
  write(paste('[TYPE] NUCLEOTIDE 2\n[SETTINGS]\n [output] FASTA\n [randomseed] ',round(runif(1,1,100000))),out_control)
  n_datasets=n_sim
  
  #Set MODEL block
  modelset=sample(c('JC','TIM','TIMef','GTR','UNREST'),n_datasets,replace=T)
  MODEL=model_gen(modelset,out_control,max_indel_length,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound)
  
  #Set TREE block
  ID_TREE=paste(rep("t_sim",times=n_datasets),1:n_datasets,sep="")
  print("Newick")
  NEWICK=read.csv(in_newick,header=TRUE)
  NEWICK=NEWICK[,2]
  print("Done newick")
  write.table(data.frame('[TREE]',ID_TREE,NEWICK),out_control,append=T,quote=F,row.names=F,col.names =F)
  
  #Set PARTITIONS block
  PNAME=paste("p",1:n_datasets,sep="")
  write.table(data.frame('[PARTITIONS]',PNAME,"[",ID_TREE,MODEL,round(runif(n_sim,len_of_msa_lower_bound,len_of_msa_upper_bound)),"]"),out_control,append=T,quote=F,row.names=F,col.names =F)
  
  #Set EVOLVE block
  write('[EVOLVE]',out_control,append=T)
  write.table(data.frame(PNAME,1,apply(data.frame(ID_TREE),1,paste,collapse="")),out_control,append=T,quote=F,row.names=F,col.names =F)
}

indelib_gen(n_taxa,n_sim,len_of_msa_lower_bound,len_of_msa_upper_bound,indel_substitution_rate_lower_bound,indel_substitution_rate_upper_bound,max_indel_length,in_newick,out_control)
