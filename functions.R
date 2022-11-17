#reticulate::use_python('C:/Users/dmorgan/AppData/Local/Continuum/anaconda3/python.exe', required = TRUE)
reticulate::use_condaenv(condaenv = "seqwell")


library(Seurat)
library(RColorBrewer)
library(ggplot2)
library(feather)
library(dplyr)

# update: 
# seurat is renormalizing the data (since python data is already normalized)
# i'm going to implement a brute-forcey work around, for now
# analysis-wise, this requires getting rid of poisson scaling
# read in files transferred from python
# read in files transferred from python
pyImport = function(name, rawfile = 'cellsAll.feather') {
  normdata = read_feather(paste0(name, '.feather')) %>% as.data.frame()
  #rownames(data) = data[,1]
  #data = data[,-1]
  
  message('reading in raw data')
  rawdata = read_feather(rawfile) %>% as.data.frame()
  rownames(rawdata) = rawdata$Gene
  id = which(colnames(rawdata) == 'Gene')
  rawdata = rawdata[,-id]
  rawdata = rawdata[,colnames(rawdata) %in% colnames(normdata)]
  
  
  metadata = read.csv(paste0(name, '_meta.txt'), row.names = 1, stringsAsFactors = FALSE)
  seurat = CreateSeuratObject(rawdata)
  seurat@meta.data = metadata
  seurat
}

# standard Seurat processing
pyProcess = function(seurat) {
  seurat = NormalizeData(seurat) 
  seurat = FindVariableGenes(seurat, do.plot= FALSE)
  seurat = ScaleData(seurat, genes.use =seurat@var.genes, vars.to.regress = c('n_genes'), model.use = 'poisson')
  seurat = RunPCA(seurat, dims.use = seurat@var.genes, do.print = FALSE)
  seurat = RunUMAP(seurat, dims.use = 1:20)
  seurat@meta.data$UMAP1 = seurat@dr$umap@cell.embeddings[,1]
  seurat@meta.data$UMAP2 = seurat@dr$umap@cell.embeddings[,2]
  seurat
}

# trying to fix annotations in metadata to be more interpretable (may be deprecated now, idk)
fixLabel= function(seurat) {
  seurat@meta.data$temp = 'Lung'
  seurat@meta.data$temp[seurat@meta.data$loc == 'Q'] = 'Subcutaneous'
  seurat@meta.data$loc = seurat@meta.data$temp
  seurat@meta.data$temp = 'Day 7'
  seurat@meta.data$temp[seurat@meta.data$treatment == 'T'] = "Day 14 Treated"
  seurat@meta.data$temp[seurat@meta.data$treatment == 'UT'] = 'Day 14 Untreated'
  seurat@meta.data$treatment = seurat@meta.data$temp
  seurat@meta.data$ci = paste(seurat@meta.data$loc, seurat@meta.data$treatment,sep = ' ')
  seurat
}

# take the files from the exportSeurat function, assemble them into a seurat object, and process
pyToSeurat = function(name, rawfile) {
  print('reading in data')
  seurat = pyImport(name, rawfile)
  print('processing data')
  seurat = pyProcess(seurat)
  seurat = fixLabel(seurat)
  print('converting to sparse format')
  seurat = MakeSparse(seurat)
  seurat
}

addUMAP = function(seurat) {
    seurat@meta.data$UMAP1 = seurat@dr$umap@cell.embeddings[,1]
    seurat@meta.data$UMAP2 = seurat@dr$umap@cell.embeddings[,2]
    seurat
} 

shuffle = function(data) {
    set.seed(1)
    data[sample(rownames(data), length(rownames(data))),]
}
# for plotting (in future should just source Andy's default plotting script)
umap_theme = theme_bw() + theme(panel.background = element_blank(),
                                panel.grid.major = element_blank(), 
                                panel.grid.minor = element_blank(),
                                axis.line = element_line(color = 'black'), 
                                panel.border=element_rect(color = 'black', size=1, fill = NA), 
                                text = element_text(family = "sans", size = 16))
geneplot= function(seurat, genes) {
    plots = c()
    for (curr in genes){
        seurat@meta.data$gene = seurat@data[curr, rownames(seurat@meta.data)]
        plots[[curr]] =  ggplot(shuffle(seurat@meta.data), aes(x = UMAP1, y = UMAP2, color = gene)) + geom_point(size = .8) +
          scale_color_viridis_c() + labs(title = curr) + guides(color = FALSE) + theme(axis.title = element_blank(), axis.text = element_blank()) + remove_grid
    }
    gg = plot_grid(plotlist = plots)
    gg
}