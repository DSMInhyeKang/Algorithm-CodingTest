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
let n = Int(readLine()!)!
var dp = [Int](repeating: 0, count: n+1)


if n == 1 {
    print(1)
} else if n == 2 {
    print(2)
} else {
    dp[1] = 1
    dp[2] = 2
    for i in 3..<n+1 {
        dp[i] = (dp[i-1] + dp[i-2]) % 10007
    }
    print(dp[n])
}
