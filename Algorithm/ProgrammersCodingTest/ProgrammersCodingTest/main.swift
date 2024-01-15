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
//func solution(_ X: String, _ Y: String) -> String {
//    var list: [String] = []
//
//    for i in (0..<10) {
//        let xCount = X.filter { String($0) == String(i) }.count
//        let yCount = Y.filter { String($0) == String(i) }.count
//        list += Array(repeating: String(i), count: min(xCount, yCount))
//    }
//
//    return list.isEmpty ? "-1" : list.filter { $0 == "0" }.count == list.count ? "0" : list.sorted(by: >).joined()
//}



// 옹알이(2)(133499)
//func solution(_ babbling: [String]) -> Int {
//    let words = [ "aya", "ye", "woo", "ma" ]
//    var result = 0
//
//    for i in babbling {
//        var bab = i
//        var valid = [String]()
//        var word = ""
//
//        for b in bab {
//            word = word + String(b)
//
//            if words.contains(word) && valid.last != word {
//                valid.append(word)
//                word = ""
//            }
//        }
//
//        if word == "" {
//            result += 1
//        }
//    }
//
//    return result
//}



// 피로도(87946)
//func solution(_ k: Int, _ dungeons: [[Int]]) -> Int {
//    var result = 0
//    var visited = [Bool](repeating: false, count: dungeons.count)
//
//
//    func dfs(_ count: Int, _ pirodo: Int){
//        if result < count{
//            result = count
//        }
//
//        for i in 0..<dungeons.count{
//            if !visited[i] && dungeons[i][0] <= pirodo{
//                visited[i] = true
//                dfs(count + 1, pirodo - dungeons[i][1])
//                visited[i] = false
//            }
//        }
//    }
//
//    dfs(0, k)
//
//    return result
//}



// 부족한 금액 계산하기(82612)
//func solution(_ price: Int, _ money: Int, _ count: Int) -> Int{
//    return max((count + 1) * count / 2 * price - money , 0)
//}



// 로또의 최고 순위와 최저 순위(77484)
//func solution(_ lottos: [Int], _ win_nums: [Int]) -> [Int] {
//    let zeroCount = lottos.filter { $0 == 0 }.count
//    let winCount: Int = win_nums.filter { lottos.contains($0) }.count
//
//    return [ min(7-winCount-zeroCount, 6), min(7-winCount, 6) ]
//}



// 약수의 개수와 덧셈(77884)
//func solution(_ left: Int, _ right: Int) -> Int {
//    return (left...right).map { i in (1...i).filter { i % $0 == 0 }.count % 2 == 0 ? i : -i }.reduce(0, +)
//}



// 숫자 문자열과 영단어(81301)
//func solution(_ s: String) -> Int {
//    var result = s
//    let numbers = ["zero","one","two","three","four","five","six","seven","eight","nine","ten"]
//
//    for i in 0..<numbers.count {
//        result = result.replacingOccurrences(of: numbers[i], with: String(i))
//    }
//
//    return Int(result)!
//}



// 체육복(42862)
//func solution(_ n: Int, _ lost: [Int], _ reserve: [Int]) -> Int {
//    let losted = lost.filter{ !reserve.contains($0) }.sorted()
//    var reserved = reserve.filter{ !lost.contains($0) }.sorted()
//
//    var result = n - losted.count
//    for lost in losted {
//        for i in 0..<reserved.count {
//            if lost == reserved[i]-1 || lost == reserved[i]+1 {
//                reserved.remove(at: i)
//                result += 1
//
//                break
//            }
//        }
//    }
//
//    return result
//}



// 키패드 누르기(67256)
//func solution(_ numbers: [Int], _ hand: String) -> String {
//    var answer = ""
//    let position = [
//        1:[0,0], 2:[0,1], 3:[0,2],
//        4:[1,0], 5:[1,1], 6:[1,2],
//        7:[2,0], 8:[2,1], 9:[2,2],
//        0:[3,1],
//    ]
//    var left = [3,0]
//    var right = [3,2]
//
//    for i in numbers {
//        if i == 1 || i == 4 || i == 7 {
//            left = position[i]!
//            answer += "L"
//        } else if i == 3 || i == 6 || i == 9 {
//            right = position[i]!
//            answer += "R"
//        } else {
//            var sizeL = abs(left[0] - position[i]![0]) + abs(left[1] - position[i]![1])
//            var sizeR = abs(right[0] - position[i]![0]) + abs(right[1] - position[i]![1])
//            if sizeL < sizeR || (sizeL == sizeR && hand == "left") {
//                left = position[i]!
//                answer += "L"
//            } else {
//                right = position[i]!
//                answer += "R"
//            }
//        }
//    }
//
//    return answer
//}



// 2016년(12901)
//func solution(_ a: Int, _ b: Int) -> String {
//    let weekday = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]
//    let dayCount = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
//    return weekday[(dayCount[0..<a - 1].reduce(0, +) + b + 4) % 7]
//}



// 짝수와 홀수(12937)
//func solution(_ num: Int) -> String {
//    return num % 2 == 0 ? "Even" : "Odd"
//}



// 정수 제곱근 판별(12934)
//func solution(_ n: Int64) -> Int64 {
//    let root = Int64(sqrt(Double(n)))
//
//    return root * root == n ? (root+1) * (root+1) : -1
//}



// 두 원 사이의 정수 쌍(181187)
//func solution(_ r1: Int, _ r2: Int) -> Int64 {
//    let dr1 = Double(r1)
//    let dr2 = Double(r2)
//
//    var result = 0.0
//    
//    for x in 1...r2 {
//        let dx = Double(x)
//        let y1 = dr1 - dx > 0 ? sqrt(pow(dr1, 2) - pow(dx, 2)) : 0
//        let y2 = sqrt(pow(dr2, 2) - pow(dx, 2))
//        
//        result += floor(y2) - (ceil(y1) - 1)
//    }
//
//    return Int64(result * 4)
//}



// 문자열을 정수로 바꾸기(12925)
//func solution(_ s: String) -> Int {
//    return Int(s)!
//}



// 소수 찾기(12921)
//func solution(_ n:Int) -> Int {
//    var check = Array(repeating: 0, count: n + 1)
//    var cnt = 0
//
//    for i in 2...n {
//        if check[i] == 0 {
//            cnt += 1
//            
//            for j in stride(from: i, to: n + 1, by: i) {
//                check[j] = 1
//            }
//        }
//    }
//
//    return cnt
//}



// 모의고사(42840)
//func solution(_ answers:[Int]) -> [Int] {
//    let answer = (
//        a: [1, 2, 3, 4, 5],
//        b: [2, 1, 2, 3, 2, 4, 2, 5],
//        c: [3, 3, 1, 1, 2, 2, 4, 4, 5, 5]
//    )
//    var point = [1:0, 2:0, 3:0]
//
//    for (i, v) in answers.enumerated() {
//        if v == answer.a[i % 5] { point[1] = point[1]! + 1 }
//        if v == answer.b[i % 8] { point[2] = point[2]! + 1 }
//        if v == answer.c[i % 10] { point[3] = point[3]! + 1 }
//    }
//
//    return point.sorted{ $0.key < $1.key }.filter{ $0.value == point.values.max() }.map{ $0.key }
//}



// 요격 시스템(181188)
//func solution(_ targets: [[Int]]) -> Int {
//    var answer = 0
//    let sorted = targets.sorted(by: { $0[1] < $1[1] })
//    
//    var end = sorted[0][1]
//
//    for target in sorted {
//        if target[0] >= end {
//            end = target[1]
//            answer += 1
//        }
//    }
//    return answer + 1
//}



// 이모티콘 할인행사(150368)
//func solution(_ users: [[Int]], _ emoticons: [Int]) -> [Int] {
//    var sale = [Int](repeating: 0, count: emoticons.count)
//    let percent = [10, 20, 30, 40]
//    var answer = [0, 0]
//    
//    func dfs(_ depth: Int) {
//        if depth == sale.count {
//            var plus = 0, money = 0
//            
//            users.forEach { user in
//                var total = 0
//                zip(sale, emoticons).forEach {
//                    if user[0] <= $0.0 { total += ($0.1 * (100 - $0.0) / 100) }
//                }
//                
//                if total >= user[1] { plus += 1; total = 0 }
//                
//                money += total
//            }
//            
//            if plus > answer[0] { answer[0] = plus; answer[1] = money }
//            else if plus == answer[0] && money > answer[1] { answer[1] = money }
//            
//            return
//        }
//        
//        percent.forEach {
//            sale[depth] = $0
//            dfs(depth+1)
//        }
//    }
//    
//    dfs(0)
//    
//    return answer
//}



// 연속된 부분 수열의 합(178870)
//func solution(_ sequence: [Int], _ k: Int) -> [Int] {
//    var bag: [[Int]] = []
//    var preSum: [Int] = [0]
//
//    for num in sequence {
//        preSum.append(preSum.last! + num)
//    }
//
//    var left = 0
//    var right = 0
//    
//    while right < preSum.count {
//        let sum = preSum[right] - preSum[left]
//        if sum == k {
//            bag.append([left, right - 1])
//            left += 1
//        } else if sum < k {
//            right += 1
//        } else {
//            left += 1
//        }
//    }
//
//    let sortedArr = bag.sorted { $0[1] - $0[0] < $1[1] - $1[0] }
//    
//    return sortedArr[0]
//}



// 마법의 엘리베이터(148653)
//func solution(_ storey: Int) -> Int {
//    var storey = storey
//    var count: Int = 0
//    
//    while storey != 0 {
//        let n = storey % 10
//        
//        if n >= 6 {
//            storey += 10 - n
//            count += 10 - n
//        } else if n == 5 && (storey / 10) % 10 >= 5 {
//            storey += 10 - n
//            count += 10 - n
//        } else {
//            count += n
//        }
//        storey /= 10
//    }
//    
//    return count
//}



// 무인도 여행(154540)
//func solution(_ maps: [String]) -> [Int] {
//    var map = [[Character]]()
//    for i in maps{
//        map.append(Array(i))
//    }
//    
//    let x_max = map.count - 1
//    let y_max = map[0].count - 1
//    var visited = [[Bool]](repeating: Array(repeating: false, count: y_max + 1 ), count: x_max + 1 )
//    
//    func dfs(_ x: Int, _ y: Int) -> Int{
//        if x < 0 || y < 0 || x > x_max || y > y_max {
//            return 0
//        }
//        if visited[x][y] == true{
//            return 0
//        }
//        if map[x][y] == "X"{
//            visited[x][y] = true
//            return 0
//        }
//        
//        visited[x][y] = true
//        var tmp = Int(String(map[x][y]))!
//        return tmp + dfs(x, y+1) + dfs(x, y - 1) + dfs(x + 1, y) + dfs(x - 1, y)
//    }
//   
//    var result = [Int]()
//
//    for i in 0...x_max{
//        for j in 0...y_max{
//            var a = dfs(i, j)
//            if a > 0{
//                result.append(a)
//            }
//        }
//    }
//    
//    return result == [] ? [-1] : result.sorted(by: <)
//}



// 택배 배달과 수거하기(150369)
//func solution(_ cap: Int, _ n: Int, _ deliveries: [Int], _ pickups: [Int]) -> Int64 {
//    var ans:Int64 = 0
//    var d = 0
//    var p = 0
//    
//    for i in stride(from: n-1, through: 0, by: -1) {
//        d += deliveries[i]
//        p += pickups[i]
//        
//        while d > 0 || p > 0 {
//            d -= cap
//            p -= cap
//            ans += Int64(( i + 1) * 2)
//        }
//    }
//    
//    return ans
//}



// 인사고과(152995)
//func solution(_ scores:[[Int]]) -> Int {
//    let wonhoScore = scores[0]
//    var maxScore = 0
//    var canWonhoGetIncentive = true
//    
//    let wonhoRank =
//    scores.sorted { $0[0] != $1[0] ? $0[0] > $1[0] : $0[1] < $1[1] }
//        .filter { score in
//            maxScore = max(maxScore, score[1])
//            if score[0] > wonhoScore[0] && score[1] > wonhoScore[1] {
//                canWonhoGetIncentive = false
//            }
//            
//            return score[1] >= maxScore &&
//            score[0] + score[1] > wonhoScore[0] + wonhoScore[1]
//        }
//        .count + 1
//    
//    return canWonhoGetIncentive ? wonhoRank : -1
//}



// 연속 펄스 부분 수열의 합(161988)
//func solution(_ sequence: [Int]) -> Int64 {
//    var sequence = sequence.enumerated().map { $1 * ($0 % 2 == 0 ? 1 : -1) }
//
//    for i in 1..<sequence.count {
//        sequence[i] += sequence[i - 1]
//    }
//
//    return Int64(max(abs(sequence.max()!), abs(sequence.min()!), abs(sequence.max()! - sequence.min()!)))
//}



// 혼자 놀기의 달인(131130)
//func solution(_ cards: [Int]) -> Int {
//    var opened = Set<Int>()
//    
//    var group = [Int]()
//    
//    for i in 0..<cards.count {
//        if opened.contains(i) { continue }
//        
//        var now = i
//        var count = 0
//        
//        while !opened.contains(now) {
//            opened.insert(now)
//        
//            count += 1
//            now = cards[now] - 1
//        }
//        
//        group.append(count)
//    }
//    
//    let sorted = group.sorted(by: >)
//    
//    return sorted.count > 1 ? sorted[0] * sorted[1] : 0
//}



// 연속 부분 수열 합의 개수(131701)
//func solution(_ elements: [Int]) -> Int {
//    var sequence = Set<Int>()
//    
//    for i in 0..<elements.count {
//        var num = 0
//
//        for offset in 0..<elements.count {
//            let validIndex = (i + offset) % elements.count
//            num += elements[validIndex]
//            sequence.insert(num)
//        }
//    }
//    return sequence.count
//}



// 시소 짝꿍(152996)
//func solution(_ weights: [Int]) -> Int64 {
//    func calculate(_ num: Int) -> Int {
//        var sum = 0
//        for i in 1..<num {
//            sum += i
//        }
//        return sum
//    }
//    var result: Int = 0
//    var arr: [Int] = Array(repeating: 0, count: 1000*4+1)
//    var multiplier = [2,4,3]
//    var divider = [1,3,2]
//    for weight in weights {
//        arr[weight] += 1
//    }
//    for i in 0..<arr.count {
//        if i < 100 || i > 1000 {
//            continue
//        }
//        if arr[i] == 0 {
//            continue
//        }
//        if arr[i] > 1 {
//            result += calculate(arr[i])
//        }
//        for j in 0..<3 {
//            if (i % divider[j] != 0) {
//                continue
//            }
//            result += arr[i] * arr[(i / divider[j]) * multiplier[j]]
//        }
//    }
//    return Int64(result)
//}



// 테이블 해시 함수(147354)
//func solution(_ data: [[Int]], _ col: Int, _ row_begin: Int, _ row_end: Int) -> Int {
//    var data = data.sorted{ $0[0] > $1[0] }.sorted { $0[col-1] < $1[col-1] }
//    var answer = 0
//    for i in stride(from: row_begin-1, through: row_end-1, by: 1) {
//        answer ^= data[i].reduce(0) { $0 + ($1 % (i+1)) }
//    }
//    
//    return answer
//}



// 가장 많이 받은 선물(258712)
//func solution(_ friends: [String], _ gifts: [String]) -> Int {
//    var answer = 0
//    
//    var dict = [String: Int]()
//    for (index, friend) in friends.enumerated() {
//        dict[friend] = index
//    }
//    
//    
//    var intArray = [Int](repeating: 0, count: friends.count)
//    var giftArrays = [[Int]](repeating: [Int](repeating: 0, count: friends.count), count: friends.count)
//    
//    for gift in gifts {
//        let strs = gift.components(separatedBy: " ")
//        giftArrays[dict[strs[0]]!][dict[strs[1]]!] += 1
//        intArray[dict[strs[0]]!] += 1
//        intArray[dict[strs[1]]!] -= 1
//    }
//    
//    for i in 0..<intArray.count {
//        var num = 0
//        for j in 0..<intArray.count where i != j {
//            if giftArrays[i][j] > giftArrays[j][i]
//                || (giftArrays[i][j] == giftArrays[j][i] && intArray[i] > intArray[j]) {
//                num += 1
//            }
//        }
//        
//        answer = max(answer, num)
//    }
//    
//    return answer
//}



// 광물 캐기(172927)
func solution(_ picks: [Int], _ minerals: [String]) -> Int {
    var picks: [Int] = picks
    var answer: Int = 0

    if picks == [0, 0, 0] { return 0 }

    var count: Int = 0
    
    if picks.reduce(0, +) * 5 > minerals.count {
        count = minerals.count
    } else {
        count = picks.reduce(0, +) * 5
    }

    var array: [[Int]] = Array(repeating: [0, 0, 0], count: 10)
    
    for i in 0..<count {
        if minerals[i] == "diamond" {
            array[i / 5][0] += 1
        }
        if minerals[i] == "iron" {
            array[i / 5][1] += 1
        }
        if minerals[i] == "stone" {
            array[i / 5][2] += 1
        }
    }

    array.sort {
        if $0[0] == $1[0] {
            if $0[1] == $1[1] {
                return $0[2] > $1[2]
            } else {
                return $0[1] > $1[1]
            }
        } else {
            return $0[0] > $1[0]
        }
    }
    
    for i in 0..<array.count {
        let (d, i, s) = (array[i][0], array[i][1], array[i][2])
        
        if picks[0] > 0 {
            picks[0] -= 1
            answer += d + i + s
        } else if picks[1] > 0 {
            picks[1] -= 1
            answer += d * 5 + i + s
        } else if picks[2] > 0 {
            picks[2] -= 1
            answer += d * 25 + i * 5 + s
        }
    }
    
    return answer
}
