import Foundation

//23/04/24
//#10814 나이순 정렬
//let n = Int(readLine()!)!
//
//var judges = [(Int, String)]()
//
//for _ in 1...n {
//    let ageAndName = (readLine()?.split(separator: " "))!
//    let age = Int(ageAndName[0])!
//    let name = String(ageAndName[1])
//
//    judges.append((age, name))
//}
//
//let sortedJudges = judges.sorted {
//    if $0.0 == $1.0 {
//        return $0.0 < $1.0
//    }
//    return $0.0 < $1.0
//}
//
//for one in sortedJudges {
//    print(one.0, one.1)
//}


//#10989 수 정렬하기3
//let cnt = Int(readLine()!)!
//var tmpArr = [Int]()
//
//for _ in 1...cnt {
//    tmpArr.append(Int(readLine()!)!)
//}
//
//tmpArr.sort() //tmpArr.sort(by: <)
//
//for i in 0..<cnt {
//    print(tmpArr[i])
//}

@available(OSX 10.15.4, *)

final class FileIO {
    private let buffer:[UInt8]
    private var index: Int = 0

    init(fileHandle: FileHandle = FileHandle.standardInput) {
        
        buffer = Array(try! fileHandle.readToEnd()!)+[UInt8(0)] // 인덱스 범위 넘어가는 것 방지
    }

    @inline(__always) private func read() -> UInt8 {
        defer { index += 1 }

        return buffer[index]
    }

    @inline(__always) func readInt() -> Int {
        var sum = 0
        var now = read()
        var isPositive = true

        while now == 10 || now == 32 { now = read() } // 공백과 줄바꿈 무시
        if now == 45 { isPositive.toggle(); now = read() } // 음수 처리
        while now >= 48, now <= 57 {
            sum = sum * 10 + Int(now-48)
            now = read()
        }

        return sum * (isPositive ? 1:-1)
    }

    @inline(__always) func readString() -> String {
        var now = read()

        while now == 10 || now == 32 { now = read() } // 공백과 줄바꿈 무시
        let beginIndex = index-1

        while now != 10, now != 32, now != 0 { now = read() }

        return String(bytes: Array(buffer[beginIndex..<(index-1)]), encoding: .ascii)!
    }

    @inline(__always) func readByteSequenceWithoutSpaceAndLineFeed() -> [UInt8] {
        var now = read()

        while now == 10 || now == 32 { now = read() } // 공백과 줄바꿈 무시
        let beginIndex = index-1

        while now != 10, now != 32, now != 0 { now = read() }

        return Array(buffer[beginIndex..<(index-1)])
    }
}

//if #available(OSX 10.15.4, *) {
//    let fIO = FileIO()
//    let n: Int = fIO.readInt()
//    var numCnt: [Int] = Array(repeating: 0, count: 10001)
//    var str = ""
//
//    Array(0..<n).forEach { _ in
//        let i: Int = fIO.readInt()
//        numCnt[i] += 1
//    }
//
//    for j in 1...10000 where numCnt[j] > 0 {
//        str.append(String(repeating: "\(j)\n", count: 0))
//    }
//    print(str)
//} else {
//    // Fallback in earlier versions
//}

//if #available(OSX 10.15.4, *) {
//    let fIO = FileIO()
//    let n: Int = fIO.readInt()
//    var tmpArr = [Int]()
//
//    for _ in 1...n {
//        tmpArr.append(fIO.readInt())
//    }
//    tmpArr.sort()
//
//    for i in 0..<n {
//        print(tmpArr[i])
//    }
//} else {
//    // Fallback in earlier versions
//}




//23/05/01
////카드(11652)
//
//let N = Int(readLine()!)!
//var tmpArr = [Int]()
//var cnt = 1
//var res_cnt = 1
//
//for _ in 1...N {
//    tmpArr.append(Int(readLine()!)!)
//}
//
//tmpArr.sort()
//var result = tmpArr[0]
//
//for i in 1..<N {
//    if tmpArr[i] == tmpArr[i-1] {
//        cnt += 1
//    } else {
//        cnt = 1
//    }
//
//    if res_cnt < cnt {
//        res_cnt = cnt
//        result = tmpArr[i]
//    }
//}
//
//print(result)



//var arr = Array(repeating: Array(repeating: 0, count: 10), count: 10)
//
//let nAndM = (readLine()?.split(separator: " "))!
//
//let n = Int(nAndM[0])
//let m = Int(nAndM[1])



//var A = [[Int]]()
//
//let nAndM = (readLine()?.split(separator: " "))!
//
//let n = Int(nAndM[0])!
//let m = Int(nAndM[1])!
//
//for i in 1...m {
//    let input = (readLine()?.split(separator: " "))!.map { Int($0)! }
//    let N = input[0], M = input[1]
//
//}

// 인접행렬
//var a: [[Int]] = []
//
//if let input = readLine() {
//    let inputs = input.split(separator: " ").map { Int($0)! }
//    let n = inputs[0]
//    let m = inputs[1]
//
//    a = Array(repeating: [], count: n + 1)
//
//    print(a)
//
//    for i in 1...n {
//        a[i] = []
//    }
//
//    for _ in 0..<m {
//        if let input = readLine() {
//            let inputs = input.split(separator: " ").map { Int($0)! }
//            let u = inputs[0]
//            let v = inputs[1]
//
//            a[u].append(v)
//            a[v].append(u)
//        }
//    }
//
//    for i in 1...n {
//        print("a[\(i)]", terminator: " ")
//        for j in 0..<a[i].count {
//            print(a[i][j], terminator: " ")
//        }
//        print()
//    }
//}



//인접 리스트
//let input = readLine()!.split(separator: " ").map { Int($0)! }
//let n = input[0]
//let m = input[1]
//
//var a = [[Int]](repeating: [], count: n + 1)
//
//for _ in 0..<m {
//    let edge = readLine()!.split(separator: " ").map { Int($0)! }
//    let u = edge[0]
//    let v = edge[1]
//    a[u].append(v)
//    a[v].append(u)
//}
//
//for i in 1...n {
//    print("a[\(i)]", terminator: " ")
//    for j in 0..<a[i].count {
//        print(a[i][j], terminator: " ")
//    }
//    print()
//}


//func dfs(_ x: Int) {
//    var check = [Bool](repeating: false, count: a.count)
//    check[x] = true
//    print("\(x) ", terminator: "")
//
//    for i in 0..<a[x].count {
//        let y = a[x][i]
//        if !check[y] {
//            dfs(y)
//        }
//    }
//}
//
//var a: [[Int]] = []
//
//dfs(0)


//// #1260 DFS와 BFS
//var graph: [[Int]] = []
//var check: [Bool] = []
//
//func dfs(_ node: Int) {
//    var s: [(Int, Int)] = []
//    s.append((node, 0))
//    check[node] = true
//    print("\(node) ", terminator: "")
//
//    while !s.isEmpty {
//        let node = s.last!.0
//        let start = s.last!.1
//        s.removeLast()
//
//        for i in start..<graph[node].count {
//            let next = graph[node][i]
//            if !check[next] {
//                print("\(next) ", terminator: "")
//                check[next] = true
//                s.append((node, i + 1))
//                s.append((next, 0))
//                break
//            }
//        }
//    }
//}
//
//func bfs(_ start: Int) {
//    var q = [Int]()
//    check = [Bool](repeating: false, count: graph.count)
//    check[start] = true
//    q.append(start)
//
//    while !q.isEmpty {
//        let node = q.removeFirst()
//        print("\(node) ", terminator: "")
//
//        for next in graph[node] {
//            if !check[next] {
//                check[next] = true
//                q.append(next)
//            }
//        }
//    }
//}
//
//// Usage example:
//let input = readLine()!.split(separator: " ").map { Int($0)! }
//let n = input[0]
//let m = input[1]
//let start = input[2]
//
//graph = [[Int]](repeating: [], count: n + 1)
//check = [Bool](repeating: false, count: n + 1)
//
//for _ in 0..<m {
//    let edge = readLine()!.split(separator: " ").map { Int($0)! }
//    let u = edge[0]
//    let v = edge[1]
//    graph[u].append(v)
//    graph[v].append(u)
//}
//
//for i in 1...n {
//    graph[i].sort()
//}
//
//dfs(start)
//print()
//bfs(start)
//print()


//// #11724 연결 요소의 개수
//let input = readLine()!.split(separator: " ").map { Int($0)! }
//let n = input[0]
//let m = input[1]
//
//var graph = Array(repeating: [], count: n + 1)
//var visited = Array(repeating: false, count: n + 1)
//
//var result = 0
//var depth = 0
//
//for _ in 0..<m {
//    let uAndV = readLine()!.split(separator: " ").map { Int($0)! }
//    let u = uAndV[0]
//    let v = uAndV[1]
//
//    graph[u].append(v)
//    graph[v].append(u)
//}
//
//func dfs(_ start: Int, _ depth: Int) {
//    visited[start] = true
//
//    for i in 0..<graph[start].count {
//        let next = graph[start][i]
//
//        if visited[next as! Int] == false {
//            dfs(next as! Int, depth + 1)
//        }
//    }
//}
//
//for j in 1..<n + 1 {
//    if !visited[j] {
//        if graph[j].isEmpty {
//            result += 1
//            visited[j] = true
//        } else {
//            dfs(j, 0)
//            result += 1
//        }
//    }
//}
//
//print(result)



////#1707 이분 그래프
//func dfs(_ a: inout [[Int]], _ color: inout [Int], _ x: Int, _ c: Int) {
//    color[x] = c
//
//    for y in a[x] {
//        if color[y] == 0 {
//            dfs(&a, &color, y, 3 - c)
//        }
//    }
//}
//
//let t = Int(readLine()!)!
//
//for _ in 0..<t {
//    let input = readLine()!.split(separator: " ").map { Int($0)! }
//    let n = input[0]
//    let m = input[1]
//
//    var a: [[Int]] = Array(repeating: [], count: n + 1)
//
//    for i in 1...n {
//        a[i] = [Int]()
//    }
//
//    for _ in 0..<m {
//        let edge = readLine()!.split(separator: " ").map { Int($0)! }
//        let u = edge[0]
//        let v = edge[1]
//        a[u].append(v)
//        a[v].append(u)
//    }
//
//    var color: [Int] = Array(repeating: 0, count: n + 1)
//    var ok = true
//
//    for i in 1...n {
//        if color[i] == 0 {
//            dfs(&a, &color, i, 1)
//        }
//    }
//
//    for i in 1...n {
//        for j in a[i] {
//            if color[i] == color[j] {
//                ok = false
//            }
//        }
//    }
//
//    if ok {
//        print("YES")
//    } else {
//        print("NO")
//    }
//}



//// #10451 순열 사이클
//let T = Int(readLine()!)!
//var graph = [Int]()
//var visited = [Bool]()
//
//func dfs(start: Int) {
//    var index = 0
//    var queue = [start]
//
//    while index < queue.count {
//        let node = queue[index]
//
//        if !visited[node] {
//            visited[node] = true
//            queue.append(graph[node])
//        }
//
//        index += 1
//    }
//}
//
//for _ in 0..<T {
//    let N = Int(readLine()!)!
//    let nums = readLine()!.split(separator: " ").map { Int($0)! }
//
//    graph = Array(repeating: 0, count: N+1)
//    visited = Array(repeating: false, count: N+1)
//    var result = 0
//
//    for j in 0..<nums.count {
//        graph[j+1] = nums[j]
//    }
//
//    for i in 1...N {
//        if !visited[i] {
//            dfs(start: i)
//            result += 1
//        }
//    }
//
//    print(result)
//}



//// #2331 반복수열
//let input = readLine()!.split(separator: " ").map { Int($0)! }
//let A = input[0]
//let P = input[1]
//
//var arr = Array(repeating: 0, count: 300000)
//var D = [Int]()
//
//func splitNums(_ A: Int, _ P: Int) -> Int {
//    var tmp = 1
//    for _ in 0..<P {
//        tmp *= A
//    }
//    return tmp
//}
//
//func getResult(_ a: Int, _ p: Int) -> Int {
//    var result = 0
//    var tempA = a
//
//    while(a > 0) {
//        result += splitNums(tempA%10, p)
//        tempA /= 10
//    }
//    return result
//}
//
//var i = 1
//arr[A] = 1
//
//while(true) {
//    var tmp = getResult(A, P)
//
//    if arr[tmp] == 0 {
//        arr[tmp] = i
//        i += 1
//    } else {
//        print(arr[tmp]-1)
//    }
//}



// 트리 순회(1991)
//let count = Int(readLine()!)!
//var results: [String] = ["", "", ""]
//var tree: [String:[String]] = [:]
//
//for _ in 0..<count {
//    let input = readLine()!.split { $0 == " " }.map { String($0) }
//    tree.updateValue([input[1], input[2]], forKey: input[0])
//}
//
//func dfs(_ node: String) {
//    if node == "." {
//        return
//    }
//
//    results[0] += node
//    dfs(tree[node]![0])
//    results[1] += node
//    dfs(tree[node]![1])
//    results[2] += node
//}
//
//dfs("A")
//
//for result in results {
//    print(result)
//}


//class Tree {
//    var left: String
//    var right: String
//
//    init(_ left: String, _ right: String) {
//        self.left = left
//        self.right = right
//    }
//}
//
//var tree: [String: Tree] = [:]
//var S1 = "", S2 = "", S3 = ""
//
//for _ in 0..<Int(readLine()!)! {
//    let N = readLine()!.split{ $0 == " " }.map{ String($0) }
//    tree[N[0]] = Tree(N[1], N[2])
//}
//
//func dfs(_ node: String) {
//    if node == "." {
//        return
//    }
//
//    S1 += node
//    dfs(tree[node]!.left)
//    S2 += node
//    dfs(tree[node]!.right)
//    S3 += node
//}
//
//dfs("A")
//print(S1)
//print(S2)
//print(S3)



// 1로 만들기(1463)
//let n = Int(readLine()!)!
//var dp = [Int](repeating: 0, count: n+1)
//
//for i in 2..<n+1 {
//    dp[i] = dp[i-1] + 1
//    if i % 3 == 0 {
//        dp[i] = min(dp[i], dp[i/3]+1)
//    }
//    if i % 2 == 0 {
//        dp[i] = min(dp[i], dp[i/2]+1)
//    }
//}
//
//print(dp[n])



// 2×n 타일링(11726)
//let n = Int(readLine()!)!
//var dp = [Int](repeating: 0, count: n+1)
//
//
//if n == 1 {
//    print(1)
//} else if n == 2 {
//    print(2)
//} else {
//    dp[1] = 1
//    dp[2] = 2
//    for i in 3..<n+1 {
//        dp[i] = (dp[i-1] + dp[i-2]) % 10007
//    }
//    print(dp[n])
//}



// 2xn 타일링(11727)
//let n = Int(readLine()!)!
//var dp = [Int](repeating: 0, count: n+1)
//
//dp[0] = 1
//dp[1] = 1
//
//for i in 2..<n+1 {
//    dp[i] = (dp[i-1] + 2*dp[i-2]) % 10007
//}
//
//print(dp[n])



// 1,2,3 더하기(9095)
//let count = Int(readLine()!) ?? 0
//var dp = [Int](repeating: 0, count: 11)
//
//dp[0] = 1
//dp[1] = 1
//dp[2] = 2
//
//for j in 3..<11 {
//    dp[j] = (dp[j-1] + dp[j-2] + dp[j-3])
//}
//
//for _ in 0..<count {
//    let n = Int(readLine()!)!
//
//    print(dp[n])
//}



// 카드 구매하기(11052)
//let n = Int(readLine()!)!
//let price = readLine()!.split(separator: " ").map{ Int($0)! }
//var dp = [Int](repeating: 0, count: n+1)
//
//
//for i in 1..<n+1 {
//    for j in 1..<i+1 {
//        dp[i] = max(dp[i], dp[i-j]+price[j-1])
//    }
//}
//
//print(dp[n])



// 카드 구매하기 2(16194)
//let n = Int(readLine()!)!
//let p = readLine()!.split(separator: " ").map { Int(String($0))! }
//var dp = [Int](repeating: 999999, count: n+1)
//
//dp[0] = 0
//
//for i in 1..<n+1 {
//    for j in 1..<i+1 {
//        dp[i] = min(dp[i], dp[i-j]+p[j-1])
//    }
//}

//print(dp[n])



// 이친수(2193)
//let n = Int(readLine()!)!
//var dp = Array(repeating: 0, count: 91)
//
//dp[1] = 1
//dp[2] = 1
//
//for i in stride(from: 3, through: n, by: 1){
//    dp[i] = dp[i - 1] + dp[i - 2]
//}
//
//print(dp[n])



// 1,2,3 더하기 5(15990)
//let T = Int(readLine()!)!
//var dp = [[Int64]]()
//
//for _ in 0...100001 {
//    var row = [Int64]()
//
//    for _ in 0...4 {
//        row.append(0)
//    }
//    
//    dp.append(row)
//}
//
//
//dp[1][1] = 1
//dp[2][2] = 1
//dp[3][1] = 1
//dp[3][2] = 1
//dp[3][3] = 1
//
//for i in stride(from: 4, through: 100000, by: 1) {
//    dp[i][1] = (dp[i-1][2] % 1000000009 + dp[i-1][3] % 1000000009) % 1000000009
//    dp[i][2] = (dp[i-2][1] % 1000000009 + dp[i-2][3] % 1000000009) % 1000000009
//    dp[i][3] = (dp[i-3][1]  % 1000000009 + dp[i-3][2] % 1000000009) % 1000000009
//}
//
//for _ in 0..<T {
//    let N = Int(readLine()!)!
//    print("\((dp[N][1] + dp[N][2] + dp[N][3]) % 1000000009)")
//}



// 쉬운 계단 수(10844)
//let N = Int(readLine()!)!
//var dp = Array(repeating: Array(repeating: 0, count: 10), count: 101)
//var sum = 0
//
//for i in 0...9 {
//    dp[1][i] = 1
//}
//
//for i in 2...100 {
//    dp[i][0] = dp[i-1][1]
//    
//    for j in 1...8 {
//        dp[i][j] = (dp[i-1][j-1] + dp[i-1][j+1]) % 1000000000
//    }
//    
//    dp[i][9] = dp[i-1][8]
//}
//
//for i in 1...9 {
//    sum = (sum + dp[N][i]) % 1000000000
//}
//
//print(sum)



// 오르막 수(11057)
//let N = Int(readLine()!)!
//var dp = Array(repeating: Array(repeating: 0, count: 10), count: N+1)
//var sol = 0
//
//for i in 0...9 {
//    dp[1][i] = 1
//}
//
//if N > 1 {
//    for i in 2...N {
//        for j in 0...9 {
//            for k in 0...j {
//                dp[i][j] = (dp[i][j]%10007 + dp[i-1][k]%10007)%10007
//            }
//        }
//    }
//}
//
//for i in 0...9 {
//    sol = (sol + dp[N][i]) % 10007
//}
//
//print(sol)



//// 스티커(9465)
//// 그리디 안 됨
//let T = Int(readLine()!)!
//
//for _ in 0..<T {
//    
//    let N = Int(readLine()!)!
//    let arr1 = readLine()!.split(separator: " ").map {Int(String($0))!}
//    let arr2 = readLine()!.split(separator: " ").map {Int(String($0))!}
//    var dp = Array(repeating: Array(repeating: 0, count: N), count: 2)
//    
//    if N == 1 {
//        print(max(arr1[0], arr2[0]))
//    } else {
//        dp[0][0] = arr1[0]
//        dp[1][0] = arr2[0]
//        dp[0][1] = dp[1][0] + arr1[1]
//        dp[1][1] = dp[0][0] + arr2[1]
//    
//        for i in 2..<N {
//            dp[0][i] = arr1[i] + max(dp[1][i-1], dp[1][i-2])
//            dp[1][i] = arr2[i] + max(dp[0][i-1], dp[0][i-2])
//        }
//        print(max(dp[0][N-1], dp[1][N-1]))
//    }
//}
//
//
//
//// 포도주 시식(2156)



// 정수 삼각형(1932) - LIS 시작
//let n = Int(readLine()!)!
//var triangle = [[Int]]()
//
//for _ in 0..<n {
//    triangle.append(readLine()!.split(separator: " ").map { Int($0)! })
//}
//
//for i in 1..<n {
//    for j in 0...i {
//        if j==0 {
//            triangle[i][j] += triangle[i-1][j]
//        }
//        else if j==i {
//            triangle[i][j] += triangle[i-1][j-1]
//        }
//        else {
//            triangle[i][j] += max(triangle[i-1][j], triangle[i-1][j-1])
//        }
//    }
//}
//
//print(triangle[n-1].max()!)



// 가장 긴 증가하는 부분 수열(11053)
//let n = Int(readLine()!)!
//let arr = readLine()!.split(separator: " ").map { Int($0)! }
//var dp = [Int]()
//
//for i in 0..<n {
//    dp.append(1)
//    for j in 0..<i {
//        if arr[j] < arr[i] && dp[i] < dp[j]+1 {
//            dp[i] = dp[j]+1
//        }
//    }
//}
//
//var answer = dp[0]
//
//for i in 0..<n {
//    if answer < dp[i] {
//        answer = dp[i]
//    }
//}
//
//print(dp.max()!)



// 가장 큰 증가 부분 수열(11055)
//let n = Int(readLine()!)!
//let arr = readLine()!.split(separator: " ").map { Int($0)! }
//var dp = [Int]()
//
//for i in 0..<n {
//    dp.append(arr[i])
//    for j in 0..<i {
//        if arr[j] < arr[i] && dp[i] < dp[j]+arr[i] {
//            dp[i] = dp[j] + arr[i]
//        }
//    }
//}
//
//var answer = 0
//for i in 0..<n {
//    if answer < dp[i] {
//        answer = dp[i]
//    }
//}
//
//print(dp.max()!)



// 가장 긴 감소하는 부분 수열(11722)
//let n = Int(readLine()!)!
//let arr = readLine()!.split(separator: " ").map { Int($0)! }
//var dp = [Int]()
//
//for i in 0..<n {
//    dp.append(1)
//    for j in 0..<i {
//        if arr[j] > arr[i] && dp[i] < dp[j] + 1 {
//            dp[i] = dp[j] + 1
//        }
//    }
//}
//
//var answer = 0
//for i in 0..<n {
//    if answer < dp[i] {
//        answer = dp[i]
//    }
//}
//
//print(dp.max()!)



// 가장 긴 바이토닉 부분 수열(11054)
//let n = Int(readLine()!)!
//let arr = readLine()!.split(separator: " ").map { Int($0)! }
//var increasing = Array(repeating: 0, count: n+1)
//var decreasing = Array(repeating: 0, count: n+1)
//
//for i in 1...n {
//    increasing[i] = 1
//    for j in 1..<i {
//        if arr[j] < arr[i] && increasing[i] < increasing[j]+1 {
//            increasing[i] = increasing[j]+1
//        }
//    }
//}
//
//for i in stride(from: n, through: 1, by: -1) {
//    decreasing[i] = 1
//    for j in stride(from: n, to: i, by: -1) {
//        if arr[j] < arr[i] && decreasing[i] < decreasing[j]+1 {
//            decreasing[i] = decreasing[j]+1
//        }
//    }
//}
//
//var answer = 0
//
//for i in 1...n {
//    if answer < increasing[i] + decreasing[i] - 1 {
//        answer = increasing[i] + decreasing[i] - 1
//    }
//}
//
//print(answer)



// 가장 긴 증가하는 부분 수열 4(14002)
//let n = Int(readLine()!)!
//let arr = readLine()!.split(separator: " ").map { Int($0)! }
//var dp = [Int]()
//
//for i in 0..<n {
//    dp.append(1)
//    for j in 0..<i {
//        if arr[j] < arr[i] && dp[i] < dp[j]+1 {
//            dp[i] = dp[j]+1
//        }
//    }
//}
//
//var answer = dp[0]
//
//for i in 0..<n {
//    if answer < dp[i] {
//        answer = dp[i]
//    }
//}
//
//print(dp.max()!)
//
//
//var LIS: [Int] = []
//var l = dp.max()!
//
//for i in (0..<n).reversed() {
//    if dp[i] == l {
//        LIS.append(arr[i])
//        l -= 1
//    }
//}
//
//print(LIS.reversed().map { String($0) }.joined(separator: " "))



// 타일 채우기(2133)
//let n = Int(readLine()!)!
//var dp = Array(repeating: 0 ,count: n+1)
//
//dp[0] = 1
//dp[2] = 3
//
//for i in stride(from: 4, through: n, by: 1){
//    dp[i] = dp[i - 2] * 3
//    for j in  stride(from: 4, through: i, by: 2){
//        dp[i] += dp[i - j] * 2
//    }
//}
//
//print(dp[n])



// N과 M(1)(15649)
//var input = readLine()!.split(separator: " ").map { Int($0)! }
//let N = input[0], M = input[1]
//
//var stack = [Int]()
//
//private func dfs() {
//    if stack.count == M {
//        print(stack.map{ String($0) }.joined(separator:" "))
//        return
//    }
//
//    for i in 1...N {
//        if !stack.contains(i) {
//            stack.append(i)
//            dfs()
//            stack.removeLast()
//        }
//
//    }
//}
//
//dfs()



// N과 M(2)(15650)
var input = readLine()!.split(separator: " ").map { Int($0)! }
let N = input[0], M = input[1]

var stack = [Int]()

private func dfs(_ start: Int) {
    if stack.count == M {
        print(stack.map{ String($0) }.joined(separator:" "))
        return
    }

    for i in start..<N+1 {
        if !stack.contains(i) {
            stack.append(i)
            dfs(i+1)
            stack.removeLast()
        }

    }
}

dfs(1)

