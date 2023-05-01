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


