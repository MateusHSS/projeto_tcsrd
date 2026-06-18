#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mesh-module.h"
#include "ns3/mesh-point-device.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"
#include "ns3/wifi-net-device.h"

#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("MeshSlaSimulation");

// Função auxiliar para quebrar strings separadas por vírgula em um conjunto de
// IDs
std::set<uint32_t> ParseFailedNodes(std::string failedNodesStr) {
  std::set<uint32_t> failedNodes;
  if (failedNodesStr == "none" || failedNodesStr == "-1" || failedNodesStr.empty()) {
    return failedNodes;
  }
  std::stringstream ss(failedNodesStr);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      failedNodes.insert(std::stoi(item));
    }
  }
  return failedNodes;
}

// Função auxiliar para converter lista de doubles separada por vírgula
std::vector<double> ParseDoubleList(std::string str, uint32_t numNodes,
                                    double defaultVal) {
  std::vector<double> vec(numNodes, defaultVal);
  std::stringstream ss(str);
  std::string item;
  uint32_t idx = 0;
  while (std::getline(ss, item, ',') && idx < numNodes) {
    if (!item.empty()) {
      vec[idx] = std::stod(item);
    }
    idx++;
  }
  return vec;
}

// Função auxiliar para parsear as coordenadas X,Y de cada nó enviadas pelo
// Python
std::vector<Vector> ParsePositions(std::string posStr, uint32_t numNodes) {
  std::vector<Vector> positions;
  std::stringstream ss(posStr);
  std::string xStr, yStr;
  while (std::getline(ss, xStr, ',') && std::getline(ss, yStr, ',')) {
    positions.push_back(Vector(std::stod(xStr), std::stod(yStr), 0.0));
  }
  // Preenche com coordenadas seguras caso haja alguma inconsistência numérica
  while (positions.size() < numNodes) {
    positions.push_back(Vector(0.0, 0.0, 0.0));
  }
  return positions;
}

// Callback fantasma para descartar pacotes em nós desativados de forma segura
bool DummyReceiveCallback(Ptr<NetDevice> device, Ptr<const Packet> packet,
                          uint16_t protocol, const Address &sender) {
  return true;
}

int main(int argc, char *argv[]) {
  uint32_t numNodes = 20;
  double simTime = 3.0; // Tempo de simulação (segundos)
  double range = 175.0; // Alcance do canal geométrico de comunicação em metros
  std::string failedNodesArg = "";
  std::string cpuValuesArg = "";
  std::string memValuesArg = "";
  std::string nodePositionsArg = "";

  CommandLine cmd(__FILE__);
  cmd.AddValue("numNodes", "Total de nós na rede Mesh", numNodes);
  cmd.AddValue("failedNodes",
               "Lista de IDs de nós derrubados separados por vírgula",
               failedNodesArg);
  cmd.AddValue("cpuValues", "Valores de CPU por nó separados por vírgula",
               cpuValuesArg);
  cmd.AddValue("memValues", "Valores de Memória por nó separados por vírgula",
               memValuesArg);
  cmd.AddValue("nodePositions", "Coordenadas X,Y de cada nó (x0,y0,x1,y1...)",
               nodePositionsArg);
  cmd.AddValue("range", "Raio de alcance da comunicação sem fio em metros",
               range);
  cmd.Parse(argc, argv);

  std::set<uint32_t> failedNodes = ParseFailedNodes(failedNodesArg);
  std::vector<double> cpus = ParseDoubleList(cpuValuesArg, numNodes, 1.0);
  std::vector<double> mems = ParseDoubleList(memValuesArg, numNodes, 1.0);
  std::vector<Vector> nodePos = ParsePositions(nodePositionsArg, numNodes);

  NodeContainer nodes;
  nodes.Create(numNodes);

  // Configuração de Mobilidade (Define a posição física exata dos nós vinda do
  // NetworkX)
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc =
      CreateObject<ListPositionAllocator>();
  for (uint32_t i = 0; i < numNodes; ++i) {
    positionAlloc->Add(nodePos[i]);
  }
  mobility.SetPositionAllocator(positionAlloc);
  mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility.Install(nodes);

  // Instalação do Stack WiFi e do Canal de Comunicação com limite físico de
  // alcance
  WifiHelper wifi;
  YansWifiPhyHelper wifiPhy;
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
  wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");

  // RangePropagationLossModel se comporta de forma idêntica ao raio geométrico
  // do NetworkX
  wifiChannel.AddPropagationLoss("ns3::RangePropagationLossModel", "MaxRange",
                                 DoubleValue(range));
  wifiPhy.SetChannel(wifiChannel.Create());

  // Define a tecnologia de rede Mesh 802.11s baseada em HWMP (Hybrid Wireless
  // Mesh Protocol)
  MeshHelper mesh = MeshHelper::Default();
  mesh.SetStackInstaller("ns3::Dot11sStack");
  NetDeviceContainer devices = mesh.Install(wifiPhy, nodes);

  // Instalação da pilha TCP/IP
  InternetStackHelper internet;
  internet.Install(nodes);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);

  // Aplicação da Injeção de Falhas e Efeitos de Variabilidade (Congestionamento
  // CPU/Memória)
  for (uint32_t i = 0; i < numNodes; ++i) {
    Ptr<Node> node = nodes.Get(i);

    if (failedNodes.find(i) != failedNodes.end()) {
      // Desativa completamente o encaminhamento IP e as interfaces de rede do
      // nó falho
      Ptr<Ipv4> nodeIpv4 = node->GetObject<Ipv4>();
      if (nodeIpv4) {
        // Preserva a interface Loopback (índice 0) para evitar travamento do protocolo IP,
        // desativando apenas as interfaces físicas reais de comunicação (índices >= 1).
        for (uint32_t iface = 1; iface < nodeIpv4->GetNInterfaces(); ++iface) {
          nodeIpv4->SetDown(iface);
          nodeIpv4->SetForwarding(iface, false);
        }
      }
      // Anula callbacks de recebimento a nível de NetDevice apenas para dispositivos Wifi/Mesh,
      // preservando o dispositivo de Loopback para evitar falhas e crashes no protocolo IP.
      for (uint32_t d = 0; d < node->GetNDevices(); ++d) {
        Ptr<NetDevice> dev = node->GetDevice(d);
        if (DynamicCast<MeshPointDevice>(dev) || DynamicCast<WifiNetDevice>(dev)) {
          dev->SetReceiveCallback(MakeCallback(&DummyReceiveCallback));
        }
      }
    } else {
      // Nó ativo: Calcula a taxa de erro agregada (PER) baseada na CPU e
      // Memória
      double cpu = cpus[i];
      double mem = mems[i];

      // Fatores de perda: quanto menor o recurso (cpu/mem), maior a perda local
      // de tráfego
      double errorRate = (1.0 - cpu) * 0.15 + (1.0 - mem) * 0.10;

      if (errorRate > 0.0) {
        Ptr<NetDevice> dev = node->GetDevice(0);
        Ptr<MeshPointDevice> mpDev = DynamicCast<MeshPointDevice>(dev);
        if (mpDev) {
          std::vector<Ptr<NetDevice>> ifaces = mpDev->GetInterfaces();
          for (auto &iface : ifaces) {
            Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice>(iface);
            if (wifiDev) {
              Ptr<WifiPhy> phy = wifiDev->GetPhy();
              Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
              em->SetAttribute("ErrorRate", DoubleValue(errorRate));
              em->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
              phy->SetPostReceptionErrorModel(em);
            }
          }
        }
      }
    }
  }

  // Instala receptores de tráfego UDP (PacketSinks) em todos os nós ativos
  uint16_t port = 9;
  PacketSinkHelper sink(
      "ns3::UdpSocketFactory",
      Address(InetSocketAddress(Ipv4Address::GetAny(), port)));
  ApplicationContainer sinkApps = sink.Install(nodes);
  sinkApps.Start(Seconds(0.5));
  sinkApps.Stop(Seconds(simTime));

  // Cria geradores de tráfego UDP (OnOff) ligando todos os pares possíveis de
  // nós ativos
  for (uint32_t i = 0; i < numNodes; ++i) {
    if (failedNodes.find(i) != failedNodes.end())
      continue;

    for (uint32_t j = 0; j < numNodes; ++j) {
      if (i == j || failedNodes.find(j) != failedNodes.end())
        continue;

      Address remoteAddr(InetSocketAddress(interfaces.GetAddress(j), port));
      OnOffHelper onoff("ns3::UdpSocketFactory", remoteAddr);
      onoff.SetAttribute(
          "OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
      onoff.SetAttribute(
          "OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));

      // Taxa moderada para evitar saturação excessiva em redes grandes
      onoff.SetAttribute("DataRate", DataRateValue(DataRate("20Kbps")));
      onoff.SetAttribute("PacketSize", UintegerValue(256));

      ApplicationContainer app = onoff.Install(nodes.Get(i));
      app.Start(Seconds(1.0));
      app.Stop(Seconds(simTime - 0.2));
    }
  }

  // Instala o monitoramento de fluxos de rede
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll();

  Simulator::Stop(Seconds(simTime));
  Simulator::Run();

  // Agrega as estatísticas do FlowMonitor
  monitor->CheckForLostPackets();
  Ptr<Ipv4FlowClassifier> classifier =
      DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
  std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

  double totalDelay = 0.0;
  uint64_t txPackets = 0;
  uint64_t rxPackets = 0;
  uint32_t activeFlows = 0;

  for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i =
           stats.begin();
       i != stats.end(); ++i) {
    txPackets += i->second.txPackets;
    rxPackets += i->second.rxPackets;
    if (i->second.rxPackets > 0) {
      totalDelay += i->second.delaySum.GetSeconds() / i->second.rxPackets;
      activeFlows++;
    }
  }

  double avgDelay = (activeFlows > 0) ? (totalDelay / activeFlows) : 0.0;
  double pdr = (txPackets > 0) ? ((double)rxPackets / txPackets) * 100.0 : 0.0;

  Simulator::Destroy();

  // Exibição em formato JSON lido diretamente pelo subprocesso Python
  std::cout << "{"
            << "\"pdr\":" << pdr << ","
            << "\"avg_delay\":" << avgDelay << ","
            << "\"tx_packets\":" << txPackets << ","
            << "\"rx_packets\":" << rxPackets << "}" << std::endl;

  return 0;
}
