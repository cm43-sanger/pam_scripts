#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.help = false
params.working_directory = null
params.manifest = null

def usage = """Usage:
  nextflow run process_manifest.nf [options] --manifest MANIFEST 
  --working_directory WORKING_DIRECTORY

Process manifest file.

Required arguments:
  --manifest MANIFEST  Path to CSV manifest (required)
  --working_directory WORKING_DIRECTORY
                       Path to working directory

Options:
  --help               Show this help message and exit
"""

if (params.help) {
    log.info usage
    exit 0
}

if (!params.manifest) {
    error "Please provide --manifest with the path to the CSV file"
}
if (!params.working_directory) {
    error "Please provide --working_directory with the path to the proposed working directory"
}

process initialize_working_directory {
    script:
    def working_directory = new File(params.working_directory)
    if (working_directory.exists()) {
        if (working_directory.isDirectory()) {
            error "Working directory already exists: ${params.working_directory}"
        }
        error "Existing non-directory at working directory path: ${params.working_directory}"
    }
    if (!working_directory.mkdirs()) {
        error "Failed to create working directory: ${params.working_directory}"
    }
    """
    """
}

def test_func(fields) {
    println "Input: ${fields}"
    def name = fields[0]
    def read1  = fields[1]
    def read2 = fields[2]
    def joiner = new java.util.StringJoiner(", ")
    joiner.add("name=${name}")
          .add("read1=${read1}")
          .add("read2=${read2}")
    return "Processed: " + joiner.toString()
}

process test_process {
    input:
    val fields

    output:
    val result

    script:
    result = test_func(fields)
    """
    """
}

workflow {
    initialize_working_directory()
    def working_directory = new File(params.working_directory)
    lines = Channel
        .fromPath(params.manifest)
        .splitText()
        .filter { it.trim() }
    fields = lines.map { line ->
        def parts = line.trim().split(',')
        if (parts.size() != 3) {
            error "Invalid line in CSV: '${line}'"
        }
        parts
    }.collect()
    fields.view()
    results = test_process(fields.flatMap { fields -> fields }).collect()
    results.view()
}
