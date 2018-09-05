(function () {
    'use strict';
  
    angular.module('checkWordApp', [])
  
    .controller('wordsCtrl', ['$scope','$http',
      function($scope,$http) {
        $scope.locale = 'vancouver';
        var step;
        console.log('start controller words');
        var all_words_notin_vocab;
        var step_size;
        $http.get("/word").then(function(result) {
            step = 0;
            step_size = 5;
            console.log('startaaa');
            console.log(result);
            all_words_notin_vocab = result.data.results
            console.log(all_words_notin_vocab);
            $scope.all_words = []
            for(var j = 0 ; j < (step + 1) * step_size ; j++){
              $scope.all_words.push(all_words_notin_vocab[j]);
            }
            console.log($scope.all_words);
            if(all_words_notin_vocab.length == 1){
                $('.subnext').hide();
            }            
            $('.subprev').hide();
        }); 

        $scope.next = function(){
          console.log('start next!');
          step++;
          console.log(step);
          if(all_words_notin_vocab.length == 1){
              $('.subnext').hide();
              $('.subprev').hide();
          }
          else{
              if( step > 0 && step<all_words_notin_vocab.length-1){
                  $('.subprev').show();
                  $('.subnext').show();
              }
              if(step == all_words_notin_vocab.length-1) {
                  $('.subnext').hide();
                  $('.subprev').show();       
              }
          }
        }
        $scope.prev = function(){
          step-=1;
          console.log(step);
          if( step > 0){
              $('.subprev').show();
              $('.subnext').show();
          }
          if(step== all_words_notin_vocab.length-1) {
              $('.subprev').show();
              $('.subnext').hide();
              
          }
          if(step==0){
              $('.subprev').hide();
              $('.subnext').show();
          }
        }
      }
    ]);
  }());