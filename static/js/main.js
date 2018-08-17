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
          // $http({
          //   method: 'POST',
          //   url: '/postContent'
          // }).then(function successCallback(response) {
          //     // this callback will be called asynchronously
          //     // when the response is available
          //   }, function errorCallback(response) {
          //     // called asynchronously if an error occurs
          //     // or server returns response with an error status.
          //   });
          $http.post('/postContent', {"content": $scope.content}).
            success(function(results) {
                console.log(str(results));
            }).
            error(function(error) {
                console.log(error);
            });
          console.log('dmdm')
        //   var req = {
        //     method: 'POST',
        //     url: '/submit',
        //     headers: {
        //       'Content-Type': undefined
        //     },
        //     data: { content: $scope.content }
        //    }
           
        //    $http(req).then(function(results){console.log(results)});
        };
      }
    ]);
  }());