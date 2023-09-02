//
//  main.swift
//  ProgrammersCodingTest
//
//  Created by 강인혜 on 2023/04/30.
//

import Foundation

// 중복된 문자 제거(120888)
//func solution(_ my_string: String) -> String {
//    var result = ""
//    for str in my_string where !result.contains(str) {
//        result.append(str)
//    }
//    return result
//}



// 개인정보 수집 유효기간(150370)
//func solution(_ today:String, _ terms:[String], _ privacies:[String]) -> [Int] {
//    var answer = [Int]()
//
//    let today = getIntValueFromDateString(date: today)
//    let expiration = terms.reduce(into: [String: Int]()) { dict, str in
//        let info = str.components(separatedBy: " ")
//        dict[info[0]] = Int(info[1])!
//    }
//
//    privacies.enumerated().forEach { (offset, privacy) in
//        let temp = privacy.components(separatedBy: " ")
//        let beginDate = getIntValueFromDateString(date: temp[0])
//        let isExpired =  beginDate + expiration[temp[1]]! * 28 <= today
//        if isExpired { answer.append(offset + 1) }
//    }
//
//    return answer
//}
//
//
//func getIntValueFromDateString(date: String) -> Int {
//    let info = date.components(separatedBy: ".").map { Int($0)! }
//    return info[0] * 12 * 28 + info[1] * 28 + info[2]
//}


//신고 결과 받기(92334)
//func solution(_ id_list: [String], _ report: [String], _ k: Int) -> [Int] {
//    var reported: [String: Int] = [:]
//    var user: [String: [String]] = [:]
//
//    for rep in Set(report) { //Set으로 중복 제거하면 시간 초과 안 뜸
//        let splited = rep.split(separator: " ").map { String($0) }
//        user[splited[0]] = (user[splited[0]] ?? []) + [splited[1]] //prosecutor
//        reported[splited[1]] = (reported[splited[1]] ?? 0) + 1 //suspect
//    }
//
//    return id_list.map { id in
//        return (user[id] ?? []).reduce(0) {
//            $0 + ((reported[$1] ?? 0) >= k ? 1 : 0)
//        }
//    }
//}


// 공원 산책(172928)
//func solution(_ park: [String], _ routes: [String]) -> [Int] {
//    var (park, x, y) = (park.map { $0.map { String($0) } }, 0, 0)
//
//    for i in 0..<park.count {
//        for j in 0..<park[0].count where park[i][j] == "S" {
//             (x, y) = (i, j)
//        }
//    }
//
//    func isIn(_ x: Int, _ y: Int) -> Bool {
//        x >= 0 && y >= 0 && x < park.count && y < park[0].count
//    }
//
//    for d in routes {
//        let d = d.split { $0 == " "}.map { String ($0) }
//        let (dir, wei) = (d[0], d[1])
//        var (ixiy, skip) = ((0, 0), false)
//
//        switch dir {
//            case "N":
//                ixiy = (-1, 0)
//            case "S":
//                ixiy = (1, 0)
//            case "W":
//                ixiy = (0, -1)
//            default:
//                ixiy = (0, 1)
//        }
//
//        var (X, Y) = (x, y)
//
//        for _ in 0..<Int(wei)! where !skip {
//            (X, Y) = (X + ixiy.0, Y + ixiy.1)
//            if !isIn(X, Y) { skip = true; break }
//            if park[X][Y] == "X" { skip = true }
//        }
//
//        if skip { continue }
//        (x, y) = (X, Y)
//    }
//
//    return [x, y]
//}


// 수박수박수박수박수박수?(12922)
//func solution(_ n:Int) -> String {
//    return (0..<n).map{($0%2==0 ? "수":"박")}.reduce("", +)
//}



// 신규 아이디 추천(72410)
//func solution(_ new_id:String) -> String {
//    var new = ""
//
//    new_id.forEach {
//        if $0.isLowercase || Int(String($0)) != nil || $0 == "." || $0 == "_" || $0 == "-" {
//            if !(new.isEmpty && $0 == ".") && !(new.last == "." && $0 == ".") {
//                new.append($0)
//            }
//        } else {
//            if $0.isUppercase {
//                new.append($0.lowercased())
//            }
//        }
//    }
//
//    if new.last == "." {
//        new.removeLast()
//    }
//
//    if new.count > 15 {
//        new.removeSubrange(new.index(after: new.index(new.startIndex, offsetBy: 14)) ..< new.endIndex)
//    } else if new.count <= 2{
//        if new.count <= 2 {
//            if new.isEmpty {
//                new = "a"
//            }
//
//            while new.count < 3 {
//                new.append(new.last!)
//            }
//        }
//    }
//
//
//    if new.last == "." {
//        new.removeLast()
//    }
//
//    return new
//}



// 달리기 경주(178871)
//func solution(_ players:[String], _ callings:[String]) -> [String] {
//    var players = players
//    var callings = callings
//    var dict = [String:Int]()
//
//    for i in 0..<players.count {
//        dict[players[i]] = i
//    }
//
//    for j in 0..<callings.count {
//        var rank = dict[callings[j]]!
//        let name  = players[rank - 1]
//        players[rank - 1] = callings[j]
//        players[rank] = name
//        dict[callings[j]]! -= 1
//        dict[players[rank]]! += 1
//    }
//
//    return players
//}



// 추억 점수(176963)
//func solution(_ name:[String], _ yearning:[Int], _ photo:[[String]]) -> [Int] {
//    let score: [String: Int] = Dictionary(uniqueKeysWithValues: zip(name, yearning))
//
//    return photo.map { $0.reduce(0) { $0 + (score[$1] ?? 0) } }
//}




// 카드 뭉치(159994)
//func solution(_ cards1:[String], _ cards2:[String], _ goal:[String]) -> String {
//    var index1:Int = 0
//    var index2:Int = 0
//
//    for i in 0..<goal.count {
//        if index1 < cards1.count && goal[i] == cards1[index1] {
//            index1 += 1
//        } else if index2 < cards2.count && goal[i] == cards2[index2] {
//            index2 += 1
//        } else {
//            return "No"
//        }
//    }
//
//    return "Yes"
//}



// 크기가 작은 부분 문자열(147355)
//func solution(_ t: String, _ p: String) -> Int {
//    let len = p.count
//    var answer = 0
//
//    for i in 0..<t.count-len+1 {
//        let startIndex = t.index(t.startIndex, offsetBy: i)
//        let endIndex = t.index(t.startIndex, offsetBy: i+len-1)
//        let range = startIndex...endIndex
//
//        if Int64(t[range])! <= Int64(p)! {
//            answer += 1
//        }
//    }
//
//    return answer
//}



// 대소문자 바꿔서 출력하기(181949)
//print(readLine()!.map { $0.isUppercase ? $0.lowercased() : $0.uppercased() }.joined())



// 덧칠하기(161989)
//func solution(_ n: Int, _ m: Int, _ section: [Int]) -> Int {
//    var section = section
//    var count = 0
//
//    while !section.isEmpty {
//        let wall = section[0]
//
//        for _ in 0..<m {
//            guard let first = section.first else { break }
//
//            if first < wall + m {
//                let _ = section.removeFirst()
//            } else {
//                break
//            }
//        }
//
//        count += 1
//    }
//    return count
//}



// 삼총사(131705)
//func solution(_ number: [Int]) -> Int {
//    var answer = 0
//
//    for i in 0..<number.count {
//        for j in i+1..<number.count {
//            for k in j+1..<number.count {
//                if number[i] + number[j] + number[k] == 0 { answer += 1 }
//            }
//        }
//    }
//
//    return answer
//}



// 성격 유형 검사하기(118666)
//func solution(_ survey: [String], _ choices: [Int]) -> String {
//    let types = ["R", "T", "C", "F", "J", "M", "A", "N"]
//    var score = Array(repeating: 0, count: types.count)
//    var result = ""
//
//    for i in 0..<survey.count {
//        if choices[i] == 4 {
//            continue
//        } else if choices[i] < 4 {
//            score[types.firstIndex(of: String(survey[i].first!))!] += (4 - choices[i])
//        } else {
//            score[types.firstIndex(of: String(survey[i].last!))!] += (choices[i] - 4)
//        }
//    }
//
//    for i in stride(from: 0, to: score.count, by: 2) {
//        if score[i] >= score[i+1] { // 삼항연산자 -> 시간 초과
//            result += types[i]
//        } else {
//            result += types[i+1]
//        }
//    }
//
//    return result
//}



// 실패율(42889)
//func solution(_ N: Int, _ stages: [Int]) -> [Int] {
//    var failure = [Int: Double]()
//    var total = Double(stages.count)
//    var countArr = Array(repeating: 0, count: N+2)
//
//    for num in stages {
//        countArr[num] += 1
//    }
//
//    for i in 1..<N+1 { //filter map 고차함수 쓰면 시간 초과 -> for문 안에 최대한 사용 자제
//        if countArr[i] == 0 {
//            failure[i] = 0.0
//        } else {
//            total = total - Double(countArr[i])
//            failure[i] = Double(countArr[i]) / total
//        }
//    }
//
//    let sortArr = failure.sorted(by: <).sorted(by: { $0.1 > $1.1 })
//    let result = sortArr.map{ $0.key }
//
//    return result
//}



// 크레인 인형뽑기 게임(64061)
//func solution(_ board: [[Int]], _ moves: [Int]) -> Int {
//    var board = board
//    var basket = [Int]()
//    var result = 0
//
//    for move in moves {
//        var y = 0
//
//        while y < board.count {
//            let doll = board[y][move-1]
//
//            if doll == 0 {
//                y += 1
//                continue
//            }
//            basket.append(doll)
//            board[y][move-1] = 0
//
//            if board[y][move-1] == 0 { break }
//        }
//
//        if (basket.count >= 2) && (basket[basket.endIndex-1] == basket[basket.endIndex-2]) {
//            basket.removeLast(2) // popLast: 옵셔널 리턴
//            result += 2
//        }
//    }
//
//    return result
//}



// 기사단원의 무기(136798)
//func solution(_ number: Int, _ limit: Int, _ power: Int) -> Int {
//    var result = [Int]()
//
//    for number in 1...number{
//        var count = 0
//
//        for i in 1...Int(sqrt(Double(number))) {
//            if number % i == 0 {
//                if(i * i) == number {
//                    count += 1
//                } else {
//                    count += 2
//                }
//            }
//        }
//
//        count = count > limit ? power : count
//        result.append(count)
//    }
//
//    return result.reduce(0){ $0 + $1 }
//}



// 콜라 문제(132267)
//func cokes(_ a: Int, _ b: Int, _ n: Int) -> Int {
//    var rest = n%a
//    var count = (n/a)*b
//
//    if a > n { return 0 }
//
//    let bottles = rest + count
//
//    return count + cokes(a, b, bottles)
//}
//
//func solution(_ a: Int, _ b: Int, _ n: Int) -> Int {
//    return cokes(a, b, n)
//}



// 과일 장수(135808)
//func solution(_ k: Int, _ m: Int, _ score: [Int]) -> Int {
//    let s = score.sorted(by: >)
//
//    return stride(from: m-1, to: score.count, by: m).reduce(0) { $0 + s[$1] * m }
//}



// 문자열 나누기(140108)
//func solution(_ s: String) -> Int {
//    var answer = 0
//    var x: Character? = nil
//    var xCount = 0
//
//    for i in s {
//        if x == nil {
//            x = i
//            xCount = 1
//            answer += 1
//            continue
//        }
//
//        xCount += x == i ? 1 : -1
//
//        if xCount == 0 {
//            x = nil
//        }
//    }
//
//    return answer
//}



// 나머지가 1이 되는 수 찾기(87389)
//func solution(_ n: Int) -> Int {
//    for num in 2...n  {
//        if n % num == 1 {
//            return num
//        }
//    }
//
//    return 1
//}



// 문자열 곱하기(181940)
//func solution(_ my_string: String, _ k: Int) -> String {
//    return String(repeating: my_string, count: k)
//}



// 홀짝에 따라 다른 값 반환하기(181935)
//func solution(_ n: Int) -> Int {
//    return n % 2 == 0 ? stride(from: 0, through: n, by: 2).map { $0 * $0 }.reduce(0, +) : stride(from: 1, through: n, by: 2).reduce(0, +)
//}



// 공배수(181936)
//func solution(_ number: Int, _ n: Int, _ m: Int) -> Int {
//    return number % n == 0 && number % m == 0 ? 1 : 0
//}



// 문자 리스트를 문자열로 반환하기(181941)
//func solution(_ arr: [String]) -> String {
//    return arr.joined()
//}



// 덧셈식 출력하기(181947)
//let n = readLine()!.components(separatedBy: [" "]).map { Int($0)! }
//let (a, b) = (n[0], n[1])
//
//print(a, "+", b, "=", a+b)


// 문자열 섞기(181942)
//func solution(_ str1:String, _ str2:String) -> String {
//    return zip(str1, str2).map { String($0) + String($1) }.joined()
//}



// 홀짝 구분하기(181944)
//let n = Int(readLine()!)!
//
//print(n, "is", n%2 == 0 ? "even" : "odd")



// n의 배수(181937)
//func solution(_ num: Int, _ n: Int) -> Int {
//    return num % n == 0 ? 1 : 0
//}



// flag에 따라 다른 값 반환하기(181933)
//func solution(_ a: Int, _ b: Int, _ flag: Bool) -> Int {
//    return flag ? a+b : a-b
//}



// 문자열 붙여서 출력하기(181946)
//let inp = readLine()!.components(separatedBy: [" "]).map { $0 }
//
//print(inp.joined())



// 조건 문자열(181934)
//func solution(_ ineq: String, _ eq: String, _ n: Int, _ m: Int) -> Int {
//    switch ineq+eq {
//    case ">=":
//        return n >= m ? 1 : 0
//    case ">!":
//        return n > m ? 1 : 0
//    case "<=":
//        return n <= m ? 1 : 0
//    case "<!":
//        return n < m ? 1 : 0
//    default:
//        return -1
//    }
//}



// 더 크게 합치기(181939)
//func solution(_ a: Int, _ b: Int) -> Int {
//    return max(Int(String(a) + String(b))!, Int(String(b) + String(a))!)
//}



// 두 수의 연산값 비교하기(181938)
//func solution(_ a: Int, _ b: Int) -> Int {
//    return max(Int(String(a) + String(b))!, 2 * a * b)
//}



// 문자열 겹쳐쓰기(181943)
//func solution(_ my_string: String, _ overwrite_string: String, _ s: Int) -> String {
//    var myString = Array(my_string)
//    myString.replaceSubrange(s...(overwrite_string.count + s - 1), with: Array(overwrite_string))
//
//    return String(myString)
////    return String(my_string.prefix(s)) + overwrite_string + String(my_string.suffix(my_string.count - overwrite_string.count - s))
//}



// 서울에서 김서방 찾기(12919)
//func solution(_ seoul: [String]) -> String {
//    return "김서방은 \(seoul.firstIndex(of: "Kim")!)에 있다"
//}



// 자릿수 더하기(12931)
//func solution(_ n: Int) -> Int {
//    var answer: Int = 0
//
//    for i in String(n) {
//        answer += Int(String(i))!
//    }
//
//    return answer
//
////    return String(n).reduce(0, { $0 + Int(String($1))! });
//}



// 정수 내림차순으로 배치하기(12933)
//func solution(_ n: Int64) -> Int64 {
//    return Int64(String(String(n).sorted(by: >)))!
//}



// 시저 암호(12926)
//func solution(_ s: String, _ n: Int) -> String {
//    let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".map { $0 }
//
//    return String(
//        s.map {
//            guard let index = alphabet.firstIndex(of: Character($0.uppercased())) else { return $0 }
//
//            let word = alphabet[(index + n) % alphabet.count]
//
//            return $0.isLowercase ? Character(word.lowercased()) : word
//        }
//    )
//}



// 문자열 내 마음대로 정렬하기(12915)
//func solution(_ strings: [String], _ n: Int) -> [String] {
//    return strings.sorted {
//        Array($0)[n] == Array($1)[n] ? $0 < $1 : Array($0)[n] < Array($1)[n]
//    }
//}



// 올바른 괄호(12909)
//func solution(_ s: String) -> Bool {
//    var opens = 0
//
//    for c in s {
//        if c == "(" {
//            opens += 1;
//        } else if c == ")" {
//            opens -= 1;
//
//            if opens < 0 {
//                return false;
//            }
//        }
//    }
//    return opens == 0;
//}



// 가장 큰 정사각형 찾기(12905)
//func solution(_ board:[[Int]]) -> Int {
//    var answer:Int = 0
//    var dp = board
//
//    for i in 0..<board.count {
//        for j in 0..<board[0].count {
//            answer = max(answer, board[i][j])
//            if i - 1 < 0 || j - 1 < 0 {
//                continue
//            }
//
//            if board[i][j] == 1 && board[i-1][j] == 1 && board[i][j-1] == 1 {
//                dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1
//                answer = max(answer, dp[i][j])
//            }
//        }
//    }
//
//    return answer * answer
//}



// 땅따먹기(12913)
//func solution(_ land: [[Int]]) -> Int {
//    var dp = land
//
//    for i in 1..<land.count {
//        for j in 0..<land[0].count {
//            for k in 0..<land[0].count {
//                if j == k { continue }
//                dp[i][j] = max(dp[i][j], dp[i-1][k] + land[i][j])
//            }
//        }
//    }
//
//    return dp.last!.max()!
//}



// 가운데 글자 가져오기(12903)
//func solution(_ s: String) -> String {
//    if s.count % 2 == 0 {
//        return String(Array(s)[(s.count/2 - 1)...(s.count/2)])
//    } else {
//        return String(Array(s)[s.count/2])
//    }
////    return String(s[String.Index(encodedOffset: (s.count-1)/2)...String.Index(encodedOffset: s.count/2)])
//}



// 다음 큰 숫자(12911)
//func solution(_ n: Int) -> Int {
//    var answer : Int = n + 1
//
//    while true {
//        if n.nonzeroBitCount == answer.nonzeroBitCount {
//            break;
//        }
//        answer += 1
//    }
//
//    return answer
//}



// 내적(70128)
//func solution(_ a: [Int], _ b: [Int]) -> Int {
//    return zip(a, b).map(*).reduce(0, +)
//}



// 수식 최대화(67257)
//enum Operation: Character {
//    case multiple = "*"
//    case plus = "+"
//    case minus = "-"
//
//    func operate(_ n1: Int64, _ n2: Int64) -> Int64 {
//        switch self {
//        case .multiple:
//            return n1 * n2
//        case .plus:
//            return n1 + n2
//        case .minus:
//            return n1 - n2
//        }
//    }
//}
//
//func operate(numbers: [Int64], opers: [Character], prior: [Operation]) -> Int64 {
//    var numbers = numbers
//    var opers = opers
//
//    for oper in prior {
//        while let index = opers.firstIndex(of: oper.rawValue) {
//            numbers[index] = oper.operate(numbers[index], numbers[index+1])
//            numbers.remove(at: index+1)
//            opers.remove(at: index)
//        }
//    }
//
//    return abs(numbers[0])
//}
//
//func solution(_ expression:String) -> Int64 {
//    let numbers = expression.components(separatedBy: ["*", "+", "-"]).map{ abs(Int64($0)!) }
//    let opers = Array(expression.filter({ Int(String($0)) == nil }))
//    let myOper = "*+-"
//    var maxValue: Int64 = 0
//
//    for first in myOper {
//        for second in myOper {
//            guard first != second else { continue }
//            let third = "*+-".filter({$0 != first && $0 != second})[0]
//            let result = operate(
//                numbers: numbers, opers: opers, prior: [
//                    Operation(rawValue: first)!,
//                    Operation(rawValue: second)!,
//                    Operation(rawValue: third)!
//                ]
//            )
//            maxValue = max(maxValue, result)
//        }
//    }
//
//    return maxValue
//}



// 튜플(64065)
//func solution(_ s: String) -> [Int] {
//    var s = s
//    var answer = [Int]()
//
//    s.removeFirst(2)
//    s.removeLast(2)
//
//    s.components(separatedBy: "},{")
//        .map { $0.components(separatedBy: ",").map { Int($0)! } }
//        .sorted { $0.count < $1.count }
//        .forEach {
//            $0.forEach {
//                if !answer.contains($0) {
//                    answer.append($0)
//                }
//            }
//    }
//
//    return answer
//}



// 3진법 뒤집기(68935)
//func solution(_ n: Int) -> Int {
//    return Int(String(String(n, radix: 3).reversed()), radix: 3)!
//}



// 오픈채팅방(42888)
//enum Commend: String {
//    case Enter  = "Enter"
//    case Leave  = "Leave"
//    case Change = "Change"
//}
//
//func solution(_ record: [String]) -> [String] {
//    var user: [String: String] = [:]
//    var result: [(message: String, uid: String)] = []
//
//    for item in record {
//        let rec: [String] = item.split(separator: " ").map(String.init)
//
//        if let commend: Commend = Commend(rawValue: rec[0]) {
//            switch commend {
//            case .Enter:
//                user[rec[1]] = rec[2]
//                result.append(("님이 들어왔습니다.", rec[1]))
//            case .Leave:
//                result.append(("님이 나갔습니다.", rec[1]))
//            case .Change: user[rec[1]] = rec[2]
//            }
//        }
//    }
//
//    let answer: [String] = result.compactMap { "\(user[$0.uid]!)\($0.message)" }
//
//    return answer
//}



// 이진 변환 반복하기(70129)
//func solution(_ s: String) -> [Int] {
//    var s = s
//    var count = 0, times = 0
//
//    while s != "1" {
//        let replaceCount = s.filter { $0 == "0" }.count
//        count += replaceCount
//
//        s = String(s.count - replaceCount, radix: 2)
//        times += 1
//    }
//
//    return [times, count]
//}



// 쿼드 압축 후 개수 세기(68936)
//var G = [[Int]]()
//var Ans = (0, 0)
//
//func DFS(_ x: Int, _ y: Int, _ len: Int) {
//    var isEqual = true
//    var target = G[x][y]
//
//    for idx in x ..< x+len {
//        for jdx in y ..< y+len {
//            if target != G[idx][jdx] {
//                isEqual = false
//                break
//            }
//        }
//
//        if !isEqual { break }
//    }
//
//    if !isEqual {
//        DFS(x, y, len/2)
//        DFS(x, y+len/2, len/2)
//        DFS(x+len/2, y, len/2)
//        DFS(x+len/2, y+len/2, len/2)
//    } else {
//        if target == 0 { Ans.0 += 1 }
//        else { Ans.1 += 1 }
//    }
//}
//
//func solution(_ arr: [[Int]]) -> [Int] {
//    G = arr
//    DFS(0, 0, arr.count)
//    return [Ans.0, Ans.1]
//}



// 후보키(42890)
//var cases = [[Int]]()
//
//func solution(_ relation: [[String]]) -> Int {
//    var candidateKey = [[Int]]()
//    var colsize = [Int]()
//
//    for i in 0..<relation[0].count {
//        colsize.append(i)
//    }
//
//    for i in 0..<colsize.count {
//        combination(n: colsize, m: i+1, current: 0, pickedArray: [])
//    }
//
//    out: for c in cases {
//        let set = Set(c)
//
//        for key in candidateKey {
//            if set.isSuperset(of: key) {
//                continue out
//            }
//        }
//
//        var rowSet = Set<Array<String>>()
//
//        for row in relation {
//            var tuple = [String]()
//
//            for i in c {
//                tuple.append(row[i])
//            }
//
//            if !rowSet.contains(tuple) {
//                rowSet.insert(tuple)
//            } else { break }
//        }
//
//        if rowSet.count == relation.count {
//            candidateKey.append(c)
//        }
//    }
//
//    return candidateKey.count
//}
//
//
//func combination(n: [Int], m: Int, current index: Int, pickedArray: [Int]) {
//    if m == 0 {
//        cases.append(pickedArray)
//    } else if index == n.count {
//        return
//    } else {
//        var newSelected = pickedArray
//
//        newSelected.append(n[index])
//        combination(n: n, m: m-1, current: index+1, pickedArray: newSelected)
//        combination(n: n, m: m, current: index+1, pickedArray: pickedArray)
//    }
//}



// 최소직사각형(86491)
//func solution(_ sizes: [[Int]]) -> Int {
//    var size = sizes
//    let count = size.count;
//    var x: [Int] = []
//    var y: [Int] = []
//
//    for i in 0..<count {
//        size[i].sort()
//        x.append(size[i][0])
//        y.append(size[i][1])
//    }
//
//    return x.max()! * y.max()!
//}



// 음양 더하기(76501)
//func solution(_ absolutes:[Int], _ signs:[Bool]) -> Int {
//    return zip(absolutes, signs)
//        .map { $1 ? $0 : -$0 }
//        .reduce(0, +)
////    return (0..<absolutes.count).map { signs[$0] ? absolutes[$0] : -absolutes[$0] }.reduce(0, +)
//}



// 둘만의 암호(155652)
//func solution(_ s: String, _ skip: String, _ index: Int) -> String {
//    let arr = "abcdefghijklmnopqrstuvwxyz".map{String($0)}.filter {!skip.contains($0) }
//
//    return s.map { arr[arr.index(arr.firstIndex(of: String($0))!, offsetBy: index) % arr.count] }.joined()
//}



// 대충 만든 자판(160586)
//func solution(_ keymap: [String], _ targets: [String]) -> [Int] {
//    var map = [Character: Int]()
//    var answer = [Int]()
//
//    for key in keymap {
//        key.enumerated().forEach {
//            if map[$0.element, default: Int.max] > $0.offset {
//                map[$0.element] = $0.offset + 1
//            }
//        }
//    }
//
//    targets.forEach {
//        var sum = 0
//        for target in $0 {
//            guard let key = map[target] else { sum = -1; break }
//            sum += key
//        }
//        answer.append(sum)
//    }
//
//    return answer
//}



// 바탕화면 정리(161990)
//func solution(_ wallpaper: [String]) -> [Int] {
//    var x_arr = [Int]()
//    var y_arr = [Int]()
//
//    for (i, paper) in wallpaper.enumerated() {
//        if paper.contains("#"){
//            for (index, w) in paper.map({String($0)}).enumerated(){
//                if w == "#" {
//                    x_arr.append(i)
//                    y_arr.append(index)
//                }
//            }
//        }
//    }
//
//    return [x_arr.min() ?? 0, y_arr.min() ?? 0, x_arr.max()!+1, y_arr.max()!+1]
//}



// 뒤에 있는 큰 수 찾기(154539)
//func solution(_ numbers: [Int]) -> [Int] {
//    var result: [Int] = Array(repeating: -1, count: numbers.count)
//    var stack: [(Int, Int)] = []
//
//    for (i, n) in numbers.enumerated() {
//        if !stack.isEmpty {
//            while !stack.isEmpty && stack.last!.1 < n {
//                result[stack.removeLast().0] = n
//            }
//        }
//
//        stack.append((i, n))
//    }
//
//    return result
//}



// 호텔 대실(155651)
//func solution(_ book_time: [[String]]) -> Int {
//    var bookTime: [(Int, Int)] = []
//    var rooms: [(Int, Int)] = []
//
//    for book in book_time {
//        let start = book[0].components(separatedBy: ":")
//        let end = book[1].components(separatedBy: ":")
//        let startTime = Int(start[0])!*60 + Int(start[1])!
//        let endTime = Int(end[0])!*60 + Int(end[1])! + 10
//        bookTime.append((startTime, endTime))
//    }
//
//    bookTime.sort(by: { $0.0 < $1.0 })
//
//    loop1: for book in bookTime {
//        for (i, room) in rooms.enumerated() {
//            if !(room.0..<room.1 ~= book.0) {
//                rooms[i] = book
//                continue loop1
//            }
//        }
//        rooms.append((book.0, book.1))
//    }
//
//    return rooms.count
//}



// 가장 가까운 같은 글자(142086)
//func solution(_ s: String) -> [Int] {
////    var dict: [Character:Int] = [:]
////    var result: [Int] = []
////
////    s.enumerated().forEach { (index, char) in
////        if let pre = dict[char] {
////            dict[char] = index
////            result += [index - pre]
////        } else {
////            result.append(-1)
////            dict[char] = index
////        }
////    }
////
////    return result
//    return s.enumerated().map { (i, c) in i - (Array(s)[0..<i].lastIndex(of: c) ?? i + 1) }
//}



// 명예의 전당(1)(138477)
//func solution(_ k: Int, _ score: [Int]) -> [Int] {
//    return (0...score.count-1).map {
//        let end = $0 < k-1 ? $0 : k-1
//
//        return Array(score[0...$0].sorted(by: >)[0...end]).last!
//    }
//}



// 햄버거 만들기(133502)
//func solution(_ ingredient: [Int]) -> Int {
//    var stack = [Int]()
//    var result = 0
//
//    for i in ingredient {
//        stack.append(i)
//
//        if stack.count < 4 { continue }
//
//        if stack.count > 3 && Array(stack.suffix(4)) == [1, 2, 3, 1] {
//            stack = Array(stack.dropLast(4))
//            result += 1
//        }
//    }
//
//    return result
//}



// 택배상자(131704)
//func solution(_ order: [Int]) -> Int {
//    var stack: [Int] = []
//    var max: Int = 0
//    var count: Int = 0
//
//    for box in order {
//        if stack.last == box {
//            count += 1
//            stack.removeLast()
//        } else if max < box {
//            count += 1
//
//            for b in max+1..<box {
//                stack.append(b)
//            }
//
//            max = box
//        } else {
//            break
//        }
//    }
//    return count
//}



// 숫자 짝꿍(131128)
func solution(_ X: String, _ Y: String) -> String {
    var list: [String] = []
    
    for i in (0..<10) {
        let xCount = X.filter { String($0) == String(i) }.count
        let yCount = Y.filter { String($0) == String(i) }.count
        list += Array(repeating: String(i), count: min(xCount, yCount))
    }
    
    return list.isEmpty ? "-1" : list.filter { $0 == "0" }.count == list.count ? "0" : list.sorted(by: >).joined()
}
