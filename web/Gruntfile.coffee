module.exports = (grunt) ->
  fs = require("fs")

  # (task: String)(input: String) => Boolean
  hasChanged = (task) -> (input) ->
    cwd  = grunt.config.get("#{task}.cwd")
    dest = grunt.config.get("#{task}.dest")
    ext  = grunt.config.get("#{task}.ext")

    output = input.replace(cwd, dest)
                  .replace(/\..+$/, ext)

    if not fs.existsSync(output)
      true
    else
      fs.statSync(input).mtime > fs.statSync(output).mtime

  grunt.initConfig
    coffee:
      compile:
        expand: true
        cwd: 'src/coffee'
        src: '**/*.coffee'
        dest: 'src/js'
        ext: '.js'

    watch:
      scripts:
        files: ['src/coffee/**/*.coffee']
        tasks: ['coffee']
        options:
          spawn: false
    eco:
      app:
        options:
          amd: true
        files: [
          expand : true
          cwd: 'src/coffee'
          src : ['**/*.eco']
          ext : '.js'
          dest: 'src/js'
        ]

  grunt.loadNpmTasks("grunt-contrib-coffee")
  grunt.loadNpmTasks("grunt-contrib-watch")
  grunt.loadNpmTasks("grunt-contrib-clean")
  grunt.loadNpmTasks("grunt-eco")

  grunt.registerTask 'default', ['coffee', 'eco']
  grunt.registerTask 'build',  ['coffee', 'eco']
