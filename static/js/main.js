(function () {
    'use strict';
  
    angular.module('nluApp', [])
  
    .controller('nluController', ['$scope','$http',
      function($scope,$http) {
        $scope.getResults = function() {
          console.log("hehe");
          console.log("data");
          console.log($scope.content);
          console.log('data');
          $http.post('/postContent', {"content": $scope.content}).
            success(function(results) {
                console.log(results.outputs);
                // $scope.a = results.outputs[0];
            }).
            error(function(error) {
                console.log(error);
            });
          console.log('dmdm');
          
        };
        $scope.pretrain = function(){
          console.log("start pretrain!!!");
          $http.post('/pretrain').
            success(function(results) {
                console.log(results);
            }).
            error(function(error) {
                console.log(error);
            });
        }
      }
    ]);
  }());